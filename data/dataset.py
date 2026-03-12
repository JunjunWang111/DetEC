import os
import pandas as pd
from torch.utils.data import Dataset

class ProteinDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # 根据split选择不同的数据集文件
        if split == 'train':
            self.data_file = os.path.join(config.data_root, 'split100.csv')
        elif split == 'val':
            self.data_file = os.path.join(config.data_root, 'Temporal-Val.csv')
        elif split == 'test_new':
            self.data_file = os.path.join(config.data_root, 'New-392.csv')
        elif split == 'test_price':
            self.data_file = os.path.join(config.data_root, 'Price-149.csv')
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # 加载数据
        self.data = pd.read_csv(self.data_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        entry = row['Entry']
        ec_labels = row['EC number'].split(';')  # 处理多个EC号的情况
        sequence = row['Sequence']
    
        main_ec = ec_labels[0]
        ec_parts = main_ec.split('.')
        while len(ec_parts) < 4:
            ec_parts.append('0')
        
        seq_length = len(sequence)
        coords = [[0.0, 0.0, 0.0] for _ in range(seq_length)]  
        
        sample = {
            'entry': entry,
            'seq': sequence, 
            'sequence': sequence,
            'ec_labels': ec_labels,
            'ec_parts': ec_parts,
            'coords': coords, 
            'structure': None, 
            'active_indices': []  
        }
        
        return sample
