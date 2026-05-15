from typing import Dict, Iterable, List, Optional, Sequence, Set

import torch


def decode_prediction_sets(outputs: Dict[str, torch.Tensor], taxonomy, threshold: float = 0.5) -> List[Set[str]]:
    batch_predictions: List[Set[str]] = []
    objectness = torch.sigmoid(outputs["objectness"])
    level4_scores = torch.sigmoid(outputs["level4"])

    for batch_index in range(objectness.size(0)):
        sample_predictions = {}
        sorted_queries = torch.argsort(objectness[batch_index], descending=True)
        used_queries = []
        for query_index in sorted_queries.tolist():
            query_score = objectness[batch_index, query_index].item()
            if query_score < threshold:
                continue
            used_queries.append(query_index)

            level1_id = int(outputs["level1"][batch_index, query_index].argmax().item())
            level2_id = int(outputs["level2"][batch_index, query_index].argmax().item())
            level3_id = int(outputs["level3"][batch_index, query_index].argmax().item())
            candidates = taxonomy.candidate_full_ids(level1_id, level2_id, level3_id)

            if candidates:
                candidate_tensor = level4_scores[batch_index, query_index, candidates]
                relative_index = int(candidate_tensor.argmax().item())
                full_id = candidates[relative_index]
                full_score = float(candidate_tensor[relative_index].item())
            else:
                full_id = int(level4_scores[batch_index, query_index].argmax().item())
                full_score = float(level4_scores[batch_index, query_index, full_id].item())

            full_ec = taxonomy.decode_full_id(full_id)
            sample_predictions[full_ec] = max(sample_predictions.get(full_ec, 0.0), query_score * full_score)

        if not sample_predictions and sorted_queries.numel() > 0:
            best_query = int(sorted_queries[0].item())
            level1_id = int(outputs["level1"][batch_index, best_query].argmax().item())
            level2_id = int(outputs["level2"][batch_index, best_query].argmax().item())
            level3_id = int(outputs["level3"][batch_index, best_query].argmax().item())
            candidates = taxonomy.candidate_full_ids(level1_id, level2_id, level3_id)
            if candidates:
                candidate_tensor = level4_scores[batch_index, best_query, candidates]
                relative_index = int(candidate_tensor.argmax().item())
                full_id = candidates[relative_index]
            else:
                full_id = int(level4_scores[batch_index, best_query].argmax().item())
            sample_predictions[taxonomy.decode_full_id(full_id)] = 1.0

        batch_predictions.append(set(sample_predictions.keys()))
    return batch_predictions


def compute_multilabel_metrics(
    pred_sets: Sequence[Set[str]],
    true_sets: Sequence[Iterable[str]],
    label_space: Optional[Iterable[str]] = None,
):
    if label_space is None:
        universe = set()
        for labels in pred_sets:
            universe.update(labels)
        for labels in true_sets:
            universe.update(labels)
    else:
        universe = set(label_space)

    tp = fp = fn = tn = 0
    for pred, truth_iter in zip(pred_sets, true_sets):
        truth = set(truth_iter)
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
        tn += len(universe - (pred | truth))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }
