from typing import Dict, List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _match_cost(sample_outputs: Dict[str, torch.Tensor], targets: List[Dict]):
    obj_cost = -torch.sigmoid(sample_outputs["objectness"]).unsqueeze(-1)

    level1_cost = -F.log_softmax(sample_outputs["level1"], dim=-1)[:, [target["level1_id"] for target in targets]]
    level2_cost = -F.log_softmax(sample_outputs["level2"], dim=-1)[:, [target["level2_id"] for target in targets]]
    level3_cost = -F.log_softmax(sample_outputs["level3"], dim=-1)[:, [target["level3_id"] for target in targets]]

    level4_probs = torch.sigmoid(sample_outputs["level4"])
    level4_cost = -torch.log(level4_probs[:, [target["full_id"] for target in targets]].clamp_min(1e-8))

    return obj_cost + level1_cost + level2_cost + level3_cost + level4_cost


def compute_set_prediction_loss(outputs: Dict[str, torch.Tensor], known_targets: List[List[Dict]], taxonomy):
    total_loss = outputs["objectness"].new_tensor(0.0)
    batch_size = outputs["objectness"].size(0)

    for batch_index in range(batch_size):
        sample_outputs = {key: value[batch_index] for key, value in outputs.items()}
        targets = [target for target in known_targets[batch_index] if target["full_id"] is not None]
        num_queries = sample_outputs["objectness"].size(0)

        objectness_target = torch.zeros(num_queries, device=sample_outputs["objectness"].device)
        sample_loss = outputs["objectness"].new_tensor(0.0)

        if targets:
            cost = _match_cost(sample_outputs, targets)
            pred_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
            matched_pred = torch.as_tensor(pred_indices, device=sample_outputs["objectness"].device, dtype=torch.long)
            objectness_target[matched_pred] = 1.0

            sample_loss = sample_loss + F.binary_cross_entropy_with_logits(
                sample_outputs["objectness"], objectness_target
            )

            for pred_index, target_index in zip(pred_indices, target_indices):
                target = targets[target_index]
                sample_loss = sample_loss + F.cross_entropy(
                    sample_outputs["level1"][pred_index].unsqueeze(0),
                    torch.tensor([target["level1_id"]], device=sample_outputs["objectness"].device),
                )
                sample_loss = sample_loss + F.cross_entropy(
                    sample_outputs["level2"][pred_index].unsqueeze(0),
                    torch.tensor([target["level2_id"]], device=sample_outputs["objectness"].device),
                )
                sample_loss = sample_loss + F.cross_entropy(
                    sample_outputs["level3"][pred_index].unsqueeze(0),
                    torch.tensor([target["level3_id"]], device=sample_outputs["objectness"].device),
                )

                target_level4 = torch.zeros(taxonomy.num_full_ecs, device=sample_outputs["objectness"].device)
                target_level4[target["full_id"]] = 1.0
                sample_loss = sample_loss + F.binary_cross_entropy_with_logits(
                    sample_outputs["level4"][pred_index],
                    target_level4,
                )
        else:
            sample_loss = sample_loss + F.binary_cross_entropy_with_logits(
                sample_outputs["objectness"], objectness_target
            )

        total_loss = total_loss + sample_loss

    return total_loss / max(batch_size, 1)
