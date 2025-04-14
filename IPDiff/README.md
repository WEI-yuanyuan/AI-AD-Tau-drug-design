# IPDiff Module
Migrated from IPDiff: https://github.com/YangLing0818/IPDiff

## Sampling from given pocket
Example of sampling from pockets/7upg_pocket1.pdb: 
```console
python IPDiff/sample_from_pocket.py --pdb_path pockets/7upg_pocket1.pdb 
```
For reference, batch size 25 is used on an NVIDIA RTX A6000. 

## Change log
- 2025/04/14 commit: 
    - graphbap/bapnet.py: line 151, changed dimension of self.FusionGraph
    - utils/evaluation/sascorer.py: replaced instances of package cPickle with pickle
    - utils/data.py: updated deprecated numpy types: np.long -> np.int64, np.bool -> bool
    - configs/*.yml: changed relative model checkpoint import to absolute when combined with sample_from_pocket.py; 
        might cause issues when running other scripts for now. 
    - sample_from_pocket.py: 