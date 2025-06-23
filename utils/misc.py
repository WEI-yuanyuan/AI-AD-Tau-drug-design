import os
import glob
from tqdm import tqdm

from openeye import oechem

def remove_dir_recursive(dir_path):
    if os.path.exists(dir_path):
        for file in os.listdir(dir_path):
            if os.path.isdir(os.path.join(dir_path, file)):
                remove_dir_recursive(os.path.join(dir_path, file))
            else:
                os.remove(os.path.join(dir_path, file))
        os.rmdir(dir_path)


def create_oeb_from_sdf(sdf_files, output_path):
    oechem.OEThrow.SetLevel(oechem.OEErrorLevel_Error)
    output_path_temp = os.path.join(os.path.dirname(output_path), f'temp.oeb')
    ofs = oechem.oemolostream(output_path_temp)

    # Show progress based on file size
    total_size = sum([os.path.getsize(file) for file in sdf_files])
    current_size = 0
    pbar = tqdm(sdf_files, desc="Processing SDF files", total=len(sdf_files))
    for sdf_file in pbar:
        # ifs = oechem.oemolistream(sdf_file)
        ifs = oechem.oemolistream()
        if ifs.open(sdf_file):  # Automatically detects .gz
            for mol in ifs.GetOEMols():
                oechem.OEWriteMolecule(ofs, mol)
            ifs.close()
        else:
            print(f"Failed to open: {sdf_file}")
        current_size += os.path.getsize(sdf_file)
        pbar.set_postfix({"Progress": f"{current_size / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB"})

    ofs.close()
    os.rename(output_path_temp, output_path)

                    
def prompt_user(logger, prompt):
    logger.info(prompt + " (y/n)")
    while True:
        user_input = input()
        if user_input == 'y':
            return True
        elif user_input == 'n':
            return False
        else:
            logger.info(f"Invalid input, please enter 'y' or 'n'")
            continue