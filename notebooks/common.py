import os
import sys
import numpy as np
import torch

root_dir = '/home/adrianchen/AI-AD-Tau-drug-design'
ipdiff_dir = root_dir + '/IPDiff'
sys.path.append(root_dir)
sys.path.append(ipdiff_dir)

# Load config
import IPDiff.utils.misc as misc
config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/sampling.yml'))
train_config = misc.load_config(os.path.join(root_dir, 'IPDiff/configs/training.yml'))
misc.seed_all(config.sample.seed)