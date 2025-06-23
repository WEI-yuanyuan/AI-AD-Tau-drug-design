import os
import sys
import gzip
import argparse

import torch
from rdkit import Chem
from openeye import oechem

# enable relative imports within IPDiff
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from similarity_search import get_similarity_search_engine
from docking import get_docking_engine
from utils.misc import remove_dir_recursive, prompt_user
from utils.fine_tuning_data_prep import DPODataCacher

sys.path.append(os.path.join(root_dir, 'IPDiff'))
from IPDiff.utils.transforms import get_atomic_number_from_index, is_aromatic_from_index
from IPDiff.utils import reconstruct
from IPDiff.utils import misc


def read_molecules_from_sdf(sdf_files, max_read=None):
    molecules = []
    for sdf_file in sdf_files:
        with gzip.open(sdf_file, 'rb') as f:  # 'rt' mode for text reading
            supplier = Chem.ForwardSDMolSupplier(f)
            for mol in supplier:
                if mol is not None:  # Some molecules might fail to parse
                    molecules.append(mol)
                    if max_read is not None and len(molecules) >= max_read:
                        return molecules
    return molecules


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Stage 1: result processing, Stage 2: similarity search')
    parser.add_argument('--lib_path', type=str, default='/local/adrianchen/ZINC20-drug')
    parser.add_argument('--server_device', type=str, default='cuda:0')
    args = parser.parse_args()
    logger = misc.get_logger('postprocess')
    
    stage = args.stage
    protein_path = os.path.join(root_dir, args.protein_path)
    
    ### Stage 1: Result processing
    
    # Logistical checks for Stage 1
    sdf_gen_files_path = os.path.join(root_dir, args.result_path, 'sdf_gen')
    if stage == 1 and os.path.exists(sdf_gen_files_path) and len(os.listdir(sdf_gen_files_path)) > 0:
        if not prompt_user(logger, 
            f"Stage 1 appears to have been completed as {sdf_gen_files_path} exists and is not empty. Are you sure you want to restart?"):
            logger.info(f"Skipping Stage 1...")
            stage = 2
        else:
            logger.info(f"Force restarting Stage 1...")
            remove_dir_recursive(sdf_gen_files_path)
    elif stage > 1 and not os.path.exists(sdf_gen_files_path):
        logger.info(f"{sdf_gen_files_path} does not exist, running Stage 1...")
        stage = 1
    os.makedirs(sdf_gen_files_path, exist_ok=True)
    
    if stage == 1:
        logger.info(f"Starting Stage 1: result processing...")
        
        # # Load configs
        # config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/sampling.yml'))
        # train_config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/training.yml'))
        # misc.seed_all(config.sample.seed)
        # logger.info(f"Config: {config}")
        
        # Load results
        result_files = [fn for fn in os.listdir(os.path.join(root_dir, args.result_path)) if fn.endswith('.pt')]
        assert len(result_files) > 0, f"Error: No result files found in {os.path.join(root_dir, args.result_path)}"
        if len(result_files) != 1:
            logger.warning(f"Warning: Multiple result files found in {os.path.join(root_dir, args.result_path)}, \
                using {result_files[0]} as the result file")
        
        result_file = result_files[0]
        result = torch.load(os.path.join(root_dir, args.result_path, result_file))
        pred_ligand_pos, pred_ligand_v = [], []
        for pos, v in zip(result['pred_ligand_pos'], result['pred_ligand_v']):
            pred_ligand_pos += pos
            pred_ligand_v += v
        logger.info(f"Loaded diffusion results from {result_file}, {len(pred_ligand_pos)} samples")

        # Reconstruct ligands
        ligand_pred = []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(pred_ligand_pos, pred_ligand_v)):
            pred_atom_type = get_atomic_number_from_index(pred_v, mode='add_aromatic')
            try:
                pred_aromatic = is_aromatic_from_index(pred_v, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                print(f'{sample_idx} failed to reconstruct')
                continue
            
            if '.' in smiles:
                print(f'{sample_idx} has "." in smiles {smiles}')
                continue
            
            sdf_file = os.path.join(sdf_gen_files_path, f'{sample_idx:04d}.sdf')
            sdf_writer = Chem.SDWriter(sdf_file)
            sdf_writer.write(mol)
            sdf_writer.close()
            ligand_pred.append(sdf_file)
        logger.info(f"Saved {len(ligand_pred)} ligands to {sdf_gen_files_path}")
        stage = 2
    else: # Skip loading; use existing temp/*.sdf files
        ligand_pred = []
        for file in os.listdir(sdf_gen_files_path):
            if file.endswith('.sdf'):
                ligand_pred.append(os.path.join(sdf_gen_files_path, file))
        logger.info(f"Identified {len(ligand_pred)} ligands from {sdf_gen_files_path}")
    
    ### Stage 2: Similarity search & Docking
    
    # Perform similarity search 
    # TODO: add multiprocessing
    logger.info(f"Setting up similarity search engine...")
    search_engine = get_similarity_search_engine(args.lib_path, type='FastROCS', server_device=args.server_device, logger=logger)
    logger.info(f"Setting up docking engine...")
    docking_engine = get_docking_engine(docking_engine_type='VinaDocking', vina_path='obabel')

    cacher = DPODataCacher()
    logger.info(f"Performing similarity search and docking...")
    for idx_ligand_pred, ligand_pred in enumerate(ligand_pred):
        results, hist = search_engine.search_topk(ligand_pred, k=10)
        
        def plot_hist(hist):
            import matplotlib.pyplot as plt
            plt.bar(range(len(hist)), hist)
            plt.savefig(f'temp/hist_{idx_ligand_pred:04d}.png')
            plt.close()
        plot_hist(hist)
        logger.info(f"Similarity search histogram: {hist}")
        
        # candidates = []
        # affinities = []
        # for idx_result, result in enumerate(results):
        #     candidate_sdf, affinity = docking_engine.dock(result, protein_path)
        #     candidate_sdf_file = os.path.join(sdf_gen_files_path, f'{idx_ligand_pred:04d}_candidate_{affinity:.3f}.sdf')
        #     with open(candidate_sdf_file, 'w') as f:
        #         f.write(candidate_sdf)
        #     candidates.append(candidate_sdf_file)
        #     affinities.append(affinity)
        #     print(f"Ligand #{idx_ligand_pred} candidate #{idx_result} affinity: {affinity:.3f}")
        # cacher.add_data([protein_path] * len(results), candidates, affinities)
    search_engine._close_server()
    
    # fine_tuning_data_path = os.path.join(root_dir, args.result_path, 'fine_tuning_data')
    # os.makedirs(fine_tuning_data_path, exist_ok=True)
    # cacher.write_to_index(
    #     os.path.join(fine_tuning_data_path, 'index.pkl'), 
    #     os.path.join(fine_tuning_data_path, 'dpo_idx.pkl'),
    #     os.path.join(fine_tuning_data_path, 'split.pt'), 
    #     os.path.join(fine_tuning_data_path, 'affinity_info.pkl'), 
    #     os.path.join(fine_tuning_data_path, 'dpo_data.pt')
    # )