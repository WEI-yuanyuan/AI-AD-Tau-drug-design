# AI-AD-Tau-drug-design
AI AD Tau drug design @UCLA

## TODO
- Reorganize code to share core functions

## IPDiff Module
Refer to README.md in IPDiff/ for instructions on sampling with IPDiff. 

## Main current scripts
- preprocess_zinc.py: 
    Create Morgan fingerprints for all files within small ligand library. 
    Originally used to accelerate Tanimoto similarity search in the large 230M library. 
    Currently not in use since library size has been reduced. 
- postprocess_results.py: 
    Read, sanitize and perform similarity search on sampled results. 
    Drug candidates in the library are saved in candidates_all.pkl or as specified by --candidates_file. 
    Example usage: 
    ```console
    python postprocess_results.py --result_file sampled_results/7upg_pocket1/sample_2025-01-01_00-00-00_099.pt --protein_path pockets/7upg_pocket1.pdb
    ```
    To avoid re-processing result files, set --use_temp to True to re-use .sdf files in temp/ folder. 

## Jupyter Notebooks
- dock_and_inspect.ipynb: current downstream of postprocess_results.py, responsible for inspecting candidates and docking. 
- pipeline.ipynb: original sketch for pipeline; later portions deprecated. Refer back for visualization and docking code. 
- play_data.ipynb: inspect CrossDocked2020 data used for training IPDiff. 
- play_result_statistics.ipynb: inspect molecular weight and logP distributions of multiple sampling result files. 
- postprocess_nohup_reader.ipynb: read and analyze nohup outputs from running postprocess_results.py. 