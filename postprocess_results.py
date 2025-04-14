import os
import sys
import numpy as np
import gzip
import pickle
import time
import argparse

import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, Descriptors
from rdkit.Chem import rdFingerprintGenerator

# enable relative imports within IPDiff
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'IPDiff'))

from IPDiff.utils.visualize import visualize_protein_ligand
import IPDiff.utils.transforms as trans
from IPDiff.utils import reconstruct
from IPDiff.utils import misc


def find_tranche(mol: Chem.Mol):
    INF = 100000
    _indexer_mol_weights = [
        200, 250, 300, 325, 350, 375, 400, 425, 450, 500, INF
    ]
    _indexer_logp = [
        -1, 0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, INF
    ]
    _indexer_tranche = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'
    ]
    
    # calculate molecular weight and logP
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    
    # find the index of the molecular weight and logP
    mol_weight_index = np.searchsorted(_indexer_mol_weights, mol_weight)
    logp_index = np.searchsorted(_indexer_logp, logp)
    
    # return the tranche index
    return _indexer_tranche[mol_weight_index] + _indexer_tranche[logp_index]


def get_molecule_by_id(index_dir, molecule_id):
    """Retrieve a specific molecule using the index"""
    # Read location from index
    with open(os.path.join(index_dir, 'molecule_locations.txt'), 'r') as f:
        for line in f:
            id_, file_, start, end = line.strip().split('\t')
            if int(id_) == molecule_id:
                with gzip.open(file_, 'rb') as mol_file:
                    mol_file.seek(int(start))
                    mol_block = mol_file.read(int(end) - int(start))
                    return Chem.MolFromMolBlock(mol_block.decode())
    return None


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

def similarity_search(query_mol, index_dir, threshold=0.7):
    """
    Perform similarity search using pre-computed fingerprints
    Returns list of (molecule_id, similarity) pairs above threshold
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    query_fp = gen.GetFingerprint(query_mol)
    results = []
    
    # Read fingerprints in chunks
    chunk_size = 10000  # Adjust based on memory availability
    with open(os.path.join(index_dir, 'morgan_fp.bin'), 'rb') as fp_file:
        mol_id = 0
        while True:
            fp_bytes = fp_file.read(128 * chunk_size)  # 1024 bits = 128 bytes per fingerprint
            if not fp_bytes:
                break
            
            # Process chunk of fingerprints
            for i in range(0, len(fp_bytes), 128):
                fp = DataStructs.CreateFromBitString(fp_bytes[i:i+128].decode())
                sim = DataStructs.TanimotoSimilarity(query_fp, fp)
                if sim >= threshold:
                    results.append((mol_id, sim))
                mol_id += 1
    
    return sorted(results, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str)
    parser.add_argument('--protein_path', type=str)
    parser.add_argument('--use_temp', type=bool, default=False)
    parser.add_argument('--candidates_file', type=str, default='candidates_all.pkl')
    args = parser.parse_args()
    logger = misc.get_logger('evaluate')
    
    ### Part I: Result loading
    if not args.use_temp: # Load results and store in temp/*.sdf
        # Remove origianl temp files
        if os.path.exists('temp'):
            for file in os.listdir('temp'):
                os.remove(os.path.join('temp', file))
            os.rmdir('temp')
        os.makedirs('temp', exist_ok=True)
        
        # Load configs
        config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/sampling.yml'))
        train_config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/training.yml'))
        misc.seed_all(config.sample.seed)
        logger.info(f"Config: {config}")
        
        # Load results
        protein_path = os.path.join(root_dir, args.protein_path)
        result_file = os.path.join(root_dir, args.result_file)
        result = torch.load(result_file)
        pred_ligand_pos, pred_ligand_v = [], []
        for pos, v in zip(result['pred_ligand_pos'], result['pred_ligand_v']):
            pred_ligand_pos += pos
            pred_ligand_v += v
        logger.info(f"Loaded diffusion results from {result_file}, {len(pred_ligand_pos)} samples")

        # Reconstruct ligands
        ligand_pred = []
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(pred_ligand_pos, pred_ligand_v)):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
            try:
                pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol)
            except reconstruct.MolReconsError:
                print(f'{sample_idx} failed to reconstruct')
                continue
            
            if '.' in smiles:
                print(f'{sample_idx} has "." in smiles {smiles}')
                continue
            
            ligand_pred.append(mol)
            sdf_writer = Chem.SDWriter(os.path.join('temp', f'{sample_idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
        logger.info(f"Saved {len(ligand_pred)} ligands to sdf")
    else: # Skip loading; use existing temp/*.sdf files
        ligand_pred = []
        for file in os.listdir('temp'):
            if file.endswith('.sdf'):
                with open(os.path.join('temp', file), 'rb') as f:
                    mol = Chem.MolFromMolBlock(f.read())
                    ligand_pred.append(mol)
        logger.info(f"Loaded {len(ligand_pred)} ligands from temp")
    
    ### Part II: Similarity search
    # Load library
    lib_path = '/local/adrianchen/ZINC20-drug'
    lib_master_tranches = [_ for _ in os.listdir(lib_path) if os.path.isdir(os.path.join(lib_path, _))]
    lib_master_tranche_dict = {
        dm: [_ for _ in os.listdir(os.path.join(lib_path, dm)) if os.path.isdir(os.path.join(lib_path, dm, _))]
        for dm in lib_master_tranches
    }
    logger.info(f"Found {len(lib_master_tranches)} master tranches in ZINC library")
    
    # Find similar molecules in library for each generated ligand
    max_read = 10000 # max number of molecules to read from each master tranche to avoid memory issues
    candidates_all = []
    for idx_ligand_pred, ligand_pred in enumerate(ligand_pred):
        master_tranche = find_tranche(ligand_pred)
        lib_dir = os.path.join(lib_path, master_tranche) # For now, only search within one exact master tranche
        logger.info(f"For ligand #{idx_ligand_pred}, searching for similar molecules in {master_tranche}...")
        
        # Check if master tranche exists and has been preprocessed
        if not os.path.exists(os.path.join(lib_path, master_tranche)):
            logger.info(f"Master tranche {master_tranche} not found, skipping...")
            continue
        elif not os.path.exists(os.path.join(lib_path, master_tranche, "index/completed.txt")):
            # Currently no proprocessing is done for ZINC20-drug, so always fallback to here
            logger.info(f"Data preprocessing for {master_tranche} not completed, fallback to first {max_read} molecules...")
            
            # Read all .sdf.gz files -> molecules from master tranche
            start = time.time()
            sdf_files = [
                os.path.join(lib_dir, sub_tranche, f) 
                for sub_tranche in lib_master_tranche_dict[master_tranche]
                for f in os.listdir(os.path.join(lib_dir, sub_tranche))
                if f.endswith('.sdf.gz')
            ]
            lib_mols = read_molecules_from_sdf(sdf_files, max_read)
            
            # Calculate similarity between query molecule and all molecules in library
            def get_similarity(mol1, mol2):
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp1 = gen.GetFingerprint(mol1)
                fp2 = gen.GetFingerprint(mol2)
                return DataStructs.FingerprintSimilarity(fp1, fp2)

            # Get top-k AND above threshold similar molecules
            def get_top_similar_mols(mol, lib_mols, k=10, threshold=0.5):
                candidates = []
                for mol2 in lib_mols:
                    similarity = get_similarity(mol, mol2)
                    if similarity >= threshold:
                        candidates.append((similarity, mol2))
                candidates.sort(key=lambda x: x[0], reverse=True)
                return candidates[:k]
            
            k = 10
            thresh = 0.5
            candidates = get_top_similar_mols(ligand_pred, lib_mols, k=k, threshold=thresh)
            for idx_mol, (similarity, mol) in enumerate(candidates):
                candidates_all.append((ligand_pred, mol, similarity))
            logger.info(f"Found {len(candidates)} candidates for ligand #{idx_ligand_pred}, \
                max similarity: {f'<{thresh}' if len(candidates) == 0 else candidates[0][0]}, time: {time.time() - start}")
        else: # Use preprocessed index
            index_dir = os.path.join(lib_dir, "index")
            similar_mols = similarity_search(ligand_pred, index_dir, threshold=0.3)
            for mol_id, similarity in similar_mols:
                mol = get_molecule_by_id(index_dir, mol_id)
                candidates_all.append((ligand_pred, mol, similarity))
            if len(similar_mols) > 0:
                logger.info(f"Found {len(similar_mols)} candidates for ligand #{idx_ligand_pred}, \
                    max similarity: {similar_mols[0][1]}")
            else:
                logger.info(f"No candidates found for ligand #{idx_ligand_pred}")

        # Save candidates after similarity search for each predicted ligand
        # candidates_all: list of tuples (mol_pred, similarity, mol_lib)
        # TODO: add library ligand location to candidates_all
        with open(args.candidates_file, 'wb') as f:
            pickle.dump(candidates_all, f)