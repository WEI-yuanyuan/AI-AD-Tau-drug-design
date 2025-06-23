import re
import os
import subprocess
from typing import Union

from openbabel import pybel
from rdkit import Chem

class VinaDockingWrapper:
    def __init__(self, **kwargs):
        self.vina_path = kwargs.get('vina_path', 'vina')
    
    def dock(self, ligand: Union[str, Chem.Mol], receptor: str):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        
        if isinstance(ligand, Chem.Mol):
            ligand_sdf = 'temp/ligand.sdf'
            Chem.MolToMolFile(ligand, ligand_sdf)
        else:
            ligand_sdf = ligand
        
        l = next(pybel.readfile('sdf', ligand_sdf))
        r = next(pybel.readfile('pdb', receptor))
        l.write('pdbqt', 'temp/ligand.pdbqt', overwrite=True)
        r.write('pdbqt', 'temp/receptor.pdbqt', overwrite=True)
        
        cmd = f'{self.vina_path} -l temp/ligand.pdbqt -r temp/receptor.pdbqt -o temp/docking.pdbqt'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        stdout = result.stdout
        stderr = result.stderr
        if result.returncode != 0:
            raise RuntimeError(f"Vina docking failed: {stderr}")

        affinity_match = re.search(r"REMARK VINA RESULT:\s+([-.\d]+)", stdout)
        affinity = float(affinity_match.group(1)) if affinity_match else None
        
        docked_ligand = next(pybel.readfile('pdbqt', 'temp/docking.pdbqt'))
        sdf_block = docked_ligand.write('sdf')
        return sdf_block, affinity
        