# models/heads.py
import torch
import torch.nn as nn

class HierarchicalHead(nn.Module):
    def __init__(self, d_model, ec_levels):
        super().__init__()
        self.ec_levels = ec_levels
        self.mlp1 = nn.Linear(d_model, ec_levels[0])
        self.mlp2 = nn.Linear(d_model + ec_levels[0], ec_levels[1])
        self.mlp3 = nn.Linear(d_model + ec_levels[0] + ec_levels[1], ec_levels[2])
        self.mlp4 = nn.Linear(d_model + ec_levels[0] + ec_levels[1] + ec_levels[2], ec_levels[3])

    def forward(self, q):
        B, K, d = q.shape
        logits1 = self.mlp1(q)
        probs1 = torch.softmax(logits1, dim=-1)

        inp2 = torch.cat([q, probs1], dim=-1)
        logits2 = self.mlp2(inp2)
        probs2 = torch.softmax(logits2, dim=-1)

        inp3 = torch.cat([q, probs1, probs2], dim=-1)
        logits3 = self.mlp3(inp3)
        probs3 = torch.softmax(logits3, dim=-1)

        inp4 = torch.cat([q, probs1, probs2, probs3], dim=-1)
        logits4 = self.mlp4(inp4)
        probs4 = torch.sigmoid(logits4)

        return probs1, probs2, probs3, probs4
