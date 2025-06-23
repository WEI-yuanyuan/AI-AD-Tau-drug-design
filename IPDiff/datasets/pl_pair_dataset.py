import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, version='final', reprocess=False):
        super().__init__()
        self.raw_path = raw_path
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(self.raw_path, f'processed_database_{version}.lmdb')
        self.skipped_indices_path = os.path.join(self.raw_path, 'processed_skipped_indices.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        if reprocess or not os.path.exists(self.processed_path) or not os.path.exists(self.skipped_indices_path):
            print(f'Reprocessing specified or processed data does not exist, begin processing data')
            self._process()
        else:
            self.skipped_indices = torch.load(self.skipped_indices_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        self.skipped_indices = []
        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, *_) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    # data_prefix = '/data/work/jiaqi/binding_affinity'
                    data_prefix = self.raw_path
                    pocket_dict = PDBProtein(os.path.join(data_prefix, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(data_prefix, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    data = data.to_dict()  # avoid torch_geometric version issue
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                    self.skipped_indices.append(i)
                    continue
        db.close()
        
        print(f'Skipped {num_skipped} indices')
        torch.save(self.skipped_indices, self.skipped_indices_path)
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        if self.db is None:
            self._connect_db()
        ### Old version: 
        # key = self.keys[idx]
        ### New version with skipped indices: 
        key = str(idx).encode()
        assert key in self.keys, f'Key {key} is not valid'
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        assert data.protein_pos.size(0) > 0
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
