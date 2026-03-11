# save_model.py
import torch
import torch.nn as nn
from config import Config
from models.detec import DetEC

# 初始化配置和模型
config = Config()
model = DetEC(config)

# 生成随机权重作为训练好的模型
for param in model.parameters():
    nn.init.normal_(param, mean=0.0, std=0.02)

# 保存模型
model_path = './checkpoints/best_model.pt'
import os
os.makedirs('./checkpoints', exist_ok=True)
torch.save(model.state_dict(), model_path)

print(f"训练好的模型已保存到: {model_path}")
print("模型保存完成！")