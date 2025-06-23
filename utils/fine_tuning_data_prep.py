import pickle
import random
import torch

class DPODataCacher:
    def __init__(self):
        self.protein_file_cache = []
        self.ligand_file_cache = []
        self.affinity_cache = {}
        self.dpo_index_cache = {}
        
        self.train_test_split = 0.9
    
    def add_data(self, protein_list, ligand_list, affinity_list, win_index=0):
        assert len(protein_list) == len(ligand_list) == len(affinity_list)
        assert win_index < len(protein_list)
        
        idx_next = len(self.protein_file_cache)
        self.protein_file_cache.extend(protein_list)
        self.ligand_file_cache.extend(ligand_list)
        for ligand_file, affinity in zip(ligand_list, affinity_list):
            self.affinity_cache[ligand_file[:-4]] = affinity
        self.dpo_index_cache[idx_next + win_index] = [idx_next + i for i in range(len(protein_list)) if i != win_index]
    
    
    def write_to_index(self, pair_data_idx_path, dpo_idx_path, split_path, affinity_path):
        with open(pair_data_idx_path, 'wb') as f:
            pickle.dump(list(zip(self.protein_file_cache, self.ligand_file_cache)), f)
        with open(dpo_idx_path, 'wb') as f:
            pickle.dump(self.dpo_index_cache, f)
        
        split = self.split_train_test()
        torch.save(split, split_path)
        
        with open(affinity_path, 'wb') as f:
            pickle.dump(self.affinity_cache, f)
        
    def split_train_test(self, train_test_split=0.9, shuffle=True):
        train_size = int(len(self.protein_file_cache) * train_test_split)
        split = {'train': [], 'test': []}
        
        win_indices = list(self.dpo_index_cache.keys())
        while len(split['train']) < train_size:
            if shuffle:
                idx_win = random.choice(win_indices)
                win_indices.remove(idx_win)
            else:
                idx_win = win_indices.pop(0)
            idx_all = [idx_win] + self.dpo_index_cache[idx_win]
            split['train'].extend(idx_all)
        
        for idx_win in win_indices:
            idx_all = [idx_win] + self.dpo_index_cache[idx_win]
            split['test'].extend(idx_all)
        
        return split