import argparse
import os
import shutil
import time
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D, log_sample_categorical
from graphbap.bapnet import BAPNet
from utils.data import PDBProtein
from utils import reconstruct
from utils.evaluation import atom_num
from rdkit import Chem


def pdb_to_pocket_data(pdb_path):
    pocket_dict = PDBProtein(pdb_path).to_dict_atom()
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict={
            'element': torch.empty([0, ], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0, ], dtype=torch.long),
        }
    )
    return data


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v


def sample_diffusion_ligand_one_batch(
    model, data, batch_size=16, device='cuda:0',
    num_steps=None, pos_only=False, center_pos_mode='protein',
    sample_num_atoms='prior', net_cond=None, cond_dim=128, 
    pos_shifter=None, 
):
    batch = Batch.from_data_list([data.clone() for _ in range(batch_size)], follow_batch=FOLLOW_BATCH).to(device)

    t1 = time.time()
    with torch.no_grad():
        batch_protein = batch.protein_element_batch
        if sample_num_atoms == 'prior':
            pocket_size = atom_num.get_space_size(batch.protein_pos.detach().cpu().numpy())
            ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(batch_size)]
            batch_ligand = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(ligand_num_atoms)).to(device)
        elif sample_num_atoms == 'ref':
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
        elif sample_num_atoms == 'temp':
            # Temporary mode: number of atoms in 10~40, somewhat normal distribution
            ligand_num_atoms = []
            for _ in range(batch_size):
                num_atoms = 0
                while num_atoms < 6 or num_atoms > 20:
                    num_atoms = np.random.randn() * 7 + 13
                ligand_num_atoms.append(int(num_atoms))
            batch_ligand = torch.repeat_interleave(torch.arange(batch_size), torch.tensor(ligand_num_atoms)).to(device)
        else:
            raise ValueError

        # init ligand pos
        center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
        batch_center_pos = center_pos[batch_ligand]
        init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)

        # init ligand v
        if pos_only:
            init_ligand_v = batch.ligand_atom_feature_full
        else:
            uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)
            init_ligand_v = log_sample_categorical(uniform_logits)

        r = model.sample_diffusion(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,

            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=num_steps,
            pos_only=pos_only,
            center_pos_mode=center_pos_mode,
            net_cond=net_cond,
            cond_dim=cond_dim, 
            pos_shifter=pos_shifter, 
        )
        ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
        ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
        # unbatch pos
        ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
        ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
        pred_pos = [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                            range(batch_size)]  # num_samples * [num_atoms_i, 3]

        all_step_pos = [[] for _ in range(batch_size)]
        for p in ligand_pos_traj:  # step_i
            p_array = p.cpu().numpy().astype(np.float64)
            for k in range(batch_size):
                all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
        all_step_pos = [np.stack(step_pos) for step_pos in
                        all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
        pred_pos_traj = [p for p in all_step_pos]

        # unbatch v
        ligand_v_array = ligand_v.cpu().numpy()
        pred_v = [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(batch_size)]

        all_step_v = unbatch_v_traj(ligand_v_traj, batch_size, ligand_cum_atoms)
        pred_v_traj = [v for v in all_step_v]

        if not pos_only:
            all_step_v0 = unbatch_v_traj(ligand_v0_traj, batch_size, ligand_cum_atoms)
            pred_v0_traj = [v for v in all_step_v0]
            all_step_vt = unbatch_v_traj(ligand_vt_traj, batch_size, ligand_cum_atoms)
            pred_vt_traj = [v for v in all_step_vt]
    t2 = time.time()
    
    return pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, t2 - t1


def add_gravitational_offset(
    batched_pos: torch.Tensor, 
    protein_center: torch.Tensor, 
    protein_orth: torch.Tensor = torch.tensor([-0.191, -0.044, 0.981]), # hard-coded for 7upg pocket
    protein_offset: torch.Tensor = torch.tensor([0, 0, 4.8]), # hard-coded for 7upg pocket
    reduced_pocket_size: int = 3
) -> torch.Tensor:
    """ Add gravitational offset to the ligand positions to keep it in the pocket """
    assert batched_pos.ndim == 2 and batched_pos.shape[1] == 3
    
    protein_center = protein_center.to(batched_pos.device)
    protein_orth = protein_orth.to(batched_pos.device)
    protein_offset = protein_offset.to(batched_pos.device)
    
    dist_pos = (batched_pos - protein_center) @ protein_orth
    dist_thresh = reduced_pocket_size / 2 * (protein_offset @ protein_orth)
    forces = torch.where(
        torch.abs(torch.stack([dist_pos, dist_pos, dist_pos], axis=1)) > dist_thresh, 
        -torch.sign(dist_pos)[:, None] * protein_offset, 
        torch.zeros_like(batched_pos)
    ) # [N, 3]
    magnitudes = torch.where(
        torch.abs(dist_pos) > dist_thresh, 
        (torch.abs(dist_pos) - dist_thresh) / torch.abs(forces @ protein_orth + 1e-6), # avoid division by zero
        torch.zeros_like(dist_pos)
    )[:, None] # [N, 1]
    
    # sanity check
    assert torch.all(torch.abs((batched_pos + magnitudes * forces - protein_center) @ protein_orth) <= dist_thresh + 1e-3)
    
    return batched_pos + magnitudes * forces


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--train_config', type=str, 
                        default=os.path.join(root_dir, 'configs/training.yml'))
    parser.add_argument('--config', type=str, 
                        default=os.path.join(root_dir, 'configs/sampling.yml'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--add_gravitational_offset', type=bool, default=False)
    parser.add_argument('--reduced_pocket_size', type=float, default=3)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    train_config = misc.load_config(args.train_config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(os.path.join(root_dir, config.model.checkpoint), map_location=args.device)
    logger.info(f"Training Config: {train_config}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = train_config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # NOTE: Modify checkpoint config to grab periodic information
    ckpt['config'].model.model_type = train_config.model.model_type
    ckpt['config'].model.periodic_dir = train_config.model.periodic_dir
    ckpt['config'].model.periodic_rot = train_config.model.periodic_rot
    
    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {os.path.join(root_dir, config.model.checkpoint)}')
    
    # Load condition BAPNet
    net_cond = BAPNet(ckpt_path=os.path.join(root_dir, train_config.net_cond.ckpt_path), 
                      hidden_nf=train_config.net_cond.hidden_dim).to(args.device)

    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)
    if args.num_samples:
        config.sample.num_samples = args.num_samples
    
    # Result preparation
    from datetime import datetime
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result = {
        'data': data, 
        'pred_ligand_pos': [],
        'pred_ligand_v': [],
        'pred_ligand_pos_traj': [],
        'pred_ligand_v_traj': [],
        'time': []
    }
    result_path = os.path.join(args.result_path, os.path.basename(args.pdb_path).split('/')[-1].split('.')[0])
    result_path = os.path.join(result_path, f'results_{date}')
    os.makedirs(result_path, exist_ok=True)
    
    # Sampling
    num_batch = int(np.ceil(config.sample.num_samples / args.batch_size))
    num_total_samples = 0
    for i in tqdm(range(num_batch)):
        n_data = args.batch_size if i < num_batch - 1 else config.sample.num_samples - args.batch_size * (num_batch - 1)
        center_pos = torch.mean(data.protein_pos, dim=0)
        pos_shifter = partial(add_gravitational_offset, reduced_pocket_size=args.reduced_pocket_size) if args.add_gravitational_offset else None
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, t = sample_diffusion_ligand_one_batch( 
            model, data, 
            batch_size=n_data, 
            device=args.device, 
            num_steps=config.sample.num_steps, 
            pos_only=config.sample.pos_only, 
            center_pos_mode=config.sample.center_pos_mode, 
            sample_num_atoms='temp', #config.sample.sample_num_atoms, 
            net_cond=net_cond, 
            cond_dim=train_config.model.cond_dim, 
            pos_shifter=pos_shifter, 
        )
        result['pred_ligand_pos'].append(pred_pos)
        result['pred_ligand_v'].append(pred_v)
        result['pred_ligand_pos_traj'].append(pred_pos_traj)
        result['pred_ligand_v_traj'].append(pred_v_traj)
        result['time'].append(t)
        
        # Remove previous sample
        if os.path.exists(os.path.join(result_path, f'sample_{num_total_samples:04d}.yml')):
            os.remove(os.path.join(result_path, f'sample_{num_total_samples:04d}.yml'))
            os.remove(os.path.join(result_path, f'sample_{num_total_samples:04d}.pt'))
        # Save current sample
        num_total_samples += n_data
        shutil.copyfile(args.config, os.path.join(result_path, f'sample_{num_total_samples:04d}.yml'))
        torch.save(result, os.path.join(result_path, f'sample_{num_total_samples:04d}.pt'))
        logger.info(f'Samples up to {num_total_samples:04d} saved in {result_path}')
        
    logger.info('Sample done!')