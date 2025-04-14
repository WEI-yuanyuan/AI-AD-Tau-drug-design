import os
import gzip
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

def create_efficient_index(sdf_files, index_dir):
    """
    Create a lightweight index structure:
    1. One binary file per fingerprint type containing just the fingerprints
    2. An index file mapping molecule IDs to their locations in files
    3. No molecule storage - molecules are read from original files when needed
    """
    os.makedirs(index_dir, exist_ok=True)
    if os.path.exists(os.path.join(index_dir, 'completed.txt')):
        print(f"Index already exists for {index_dir}, skipping")
        return
    
    # If processing was not completed, clear the index files
    if os.path.exists(os.path.join(index_dir, 'molecule_locations.txt')):
        os.remove(os.path.join(index_dir, 'molecule_locations.txt'))
    if os.path.exists(os.path.join(index_dir, 'morgan_fp.bin')):
        os.remove(os.path.join(index_dir, 'morgan_fp.bin'))
        
    # Open index files
    with open(os.path.join(index_dir, 'molecule_locations.txt'), 'w') as loc_file, \
         open(os.path.join(index_dir, 'morgan_fp.bin'), 'wb') as fp_file:
        
        mol_counter = 0
        for sdf_file in sdf_files:
            file_pos = 0
            with gzip.open(sdf_file, 'rb') as f:
                content = f.read()
                # Find all molecule delimiter positions
                delimiters = [i for i in range(len(content)) if content[i:i+5] == b'$$$$\n']
                start_pos = 0
                
                for end_pos in delimiters:
                    mol_block = content[start_pos:end_pos]
                    mol = Chem.MolFromMolBlock(mol_block.decode())
                    
                    if mol is not None:
                        # Generate and write fingerprint
                        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                        fp = gen.GetFingerprint(mol)
                        fp_bytes = fp.ToBitString().encode()
                        fp_file.write(fp_bytes)
                        
                        # Write location information
                        loc_file.write(f"{mol_counter}\t{sdf_file}\t{start_pos}\t{end_pos}\n")
                        
                        mol_counter += 1
                        
                        if mol_counter != 0 and mol_counter % 10000 == 0:
                            print(f"Processed {mol_counter} molecules")
                    
                    start_pos = end_pos + 5
        
        # mark as completed
        with open(os.path.join(index_dir, 'completed.txt'), 'w') as f:
            f.write(f'Finished processing, total molecules: {mol_counter}')

if __name__ == "__main__":
    # read ZINC library
    ZINC_path = '/local/adrianchen/ZINC20'
    ZINC_lib_master_tranches = [_ for _ in os.listdir(ZINC_path) if os.path.isdir(os.path.join(ZINC_path, _))]
    ZINC_lib_master_tranche_dict = {
        dm: [_ for _ in os.listdir(os.path.join(ZINC_path, dm)) if os.path.isdir(os.path.join(ZINC_path, dm, _))]
        for dm in ZINC_lib_master_tranches
    }

    # create efficient index for each tranche
    for idx_master_tranche, master_tranche in enumerate(ZINC_lib_master_tranches):
        print(f"Processing {master_tranche}")
        sub_tranches = ZINC_lib_master_tranche_dict[master_tranche]
        sdf_files_in_tranche = [
            os.path.join(ZINC_path, master_tranche, sub_tranche, sdf_file)
            for sub_tranche in sub_tranches
            for sdf_file in os.listdir(os.path.join(ZINC_path, master_tranche, sub_tranche))
            if sdf_file.endswith('.sdf.gz')
        ]
        create_efficient_index(sdf_files_in_tranche, os.path.join(ZINC_path, master_tranche, 'index'))
        print(f"Done processing {master_tranche}")
