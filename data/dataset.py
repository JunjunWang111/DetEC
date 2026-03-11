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
        
        # 转换EC号为标签（这里简化处理，实际可能需要更复杂的编码）
        # 提取主EC号（第一个）并转换为层级结构
        main_ec = ec_labels[0]
        ec_parts = main_ec.split('.')
        while len(ec_parts) < 4:
            ec_parts.append('0')
        
        # 为模型添加必要的字段（由于没有结构信息，使用占位符）
        # 生成随机坐标作为占位符
        seq_length = len(sequence)
        coords = [[0.0, 0.0, 0.0] for _ in range(seq_length)]  # 随机坐标
        
        # 构建样本
        sample = {
            'entry': entry,
            'seq': sequence,  # 模型期望的字段名
            'sequence': sequence,
            'ec_labels': ec_labels,
            'ec_parts': ec_parts,
            'coords': coords,  # 添加坐标信息
            'structure': None,  # 结构信息（无）
            'active_indices': []  # 活性位点索引（无）
        }
        
        return sample