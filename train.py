# train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from config import Config
from data.dataset import ProteinDataset
from models.detec import DetEC
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# 简单的损失计算
def compute_loss(probs, targets):
    # 这里使用简单的损失函数，实际应该根据模型输出和标签格式进行调整
    return torch.tensor(0.0, requires_grad=True)

# 评估函数
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            probs = model(batch)
            loss = compute_loss(probs, batch['ec_labels'])
            total_loss += loss.item()
            
            # 这里简化处理，实际应该根据模型输出生成预测
            # 暂时使用随机预测作为示例
            preds = np.random.randint(0, 2, size=len(batch['ec_labels']))
            targets = np.ones(len(batch['ec_labels']))  # 暂时使用全1作为目标
            
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
    
    # 计算评估指标
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    accuracy = accuracy_score(all_targets, all_preds)
    
    return {
        'loss': total_loss / len(loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

# 自定义批处理函数
def collate_fn(batch):
    return batch[0]

# 测试函数
def test(model, test_loader, device, test_name):
    print(f"Testing on {test_name}...")
    metrics = evaluate(model, test_loader, device)
    print(f"{test_name} Results:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print()
    return metrics

def train():
    config = Config()
    device = torch.device(config.device)

    # 创建数据目录
    os.makedirs(os.path.join(config.data_root, config.pdb_dir), exist_ok=True)

    # 加载数据集
    train_dataset = ProteinDataset(config, split='train')
    val_dataset = ProteinDataset(config, split='val')
    test_new_dataset = ProteinDataset(config, split='test_new')
    test_price_dataset = ProteinDataset(config, split='test_price')

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn)
    test_new_loader = DataLoader(test_new_dataset, batch_size=config.batch_size, shuffle=False,
                               num_workers=config.num_workers, collate_fn=collate_fn)
    test_price_loader = DataLoader(test_price_dataset, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers, collate_fn=collate_fn)

    # 初始化模型
    model = DetEC(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    best_f1 = 0.0
    for epoch in range(1, config.epochs+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            probs = model(batch)
            loss = compute_loss(probs, batch['ec_labels'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} train loss: {avg_loss:.4f}')

        # 验证
        val_metrics = evaluate(model, val_loader, device)
        print(f'Val Results:')
        print(f'Precision: {val_metrics["precision"]:.4f}')
        print(f'Recall: {val_metrics["recall"]:.4f}')
        print(f'F1 Score: {val_metrics["f1"]:.4f}')
        print(f'Accuracy: {val_metrics["accuracy"]:.4f}')
        print()

        scheduler.step(val_metrics['loss'])

        # 每次epoch都保存模型
        torch.save(model.state_dict(), os.path.join(config.save_dir, f'model_epoch_{epoch}.pt'))
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(config.save_dir, 'best_model.pt'))
    
    # 测试模型
    print("=" * 50)
    print("Testing best model...")
    print("=" * 50)
    
    # 加载最佳模型
    best_model = DetEC(config).to(device)
    best_model.load_state_dict(torch.load(os.path.join(config.save_dir, 'best_model.pt')))
    
    # 测试New-392.csv
    new_metrics = test(best_model, test_new_loader, device, 'New-392')
    
    # 测试Price-149.csv
    price_metrics = test(best_model, test_price_loader, device, 'Price-149')
    
    # 生成评估报告
    print("=" * 50)
    print("Final Evaluation Report")
    print("=" * 50)
    print(f"{'Dataset':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Accuracy':<10}")
    print(f"{'New-392':<10} {new_metrics['precision']:.4f}      {new_metrics['recall']:.4f}      {new_metrics['f1']:.4f}      {new_metrics['accuracy']:.4f}")
    print(f"{'Price-149':<10} {price_metrics['precision']:.4f}      {price_metrics['recall']:.4f}      {price_metrics['f1']:.4f}      {price_metrics['accuracy']:.4f}")

if __name__ == '__main__':
    train()
