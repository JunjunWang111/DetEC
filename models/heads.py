import torch
import torch.nn as nn


class HierarchicalHead(nn.Module):
    def __init__(self, d_model, level_sizes, full_ec_size):
        super().__init__()
        self.level_sizes = level_sizes
        self.full_ec_size = full_ec_size

        self.objectness = nn.Linear(d_model, 1)
        self.mlp1 = nn.Linear(d_model, level_sizes[0])
        self.mlp2 = nn.Linear(d_model + level_sizes[0], level_sizes[1])
        self.mlp3 = nn.Linear(d_model + level_sizes[0] + level_sizes[1], level_sizes[2])
        self.mlp4 = nn.Linear(d_model + level_sizes[0] + level_sizes[1] + level_sizes[2], full_ec_size)

    def forward(self, q):
        logits_obj = self.objectness(q).squeeze(-1)

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

        return {
            "objectness": logits_obj,
            "level1": logits1,
            "level2": logits2,
            "level3": logits3,
            "level4": logits4,
        }
