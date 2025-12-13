from typing import Optional

import torch


def macro_f1(preds: torch.Tensor,
             targets: torch.Tensor,
             num_classes: Optional[int] = None,
             ignore_index: Optional[int] = None) -> torch.Tensor:
    """
    Tính Macro F1 đơn giản trên tensor preds, targets (shape [N]).
    preds: class id (long)
    targets: class id (long)
    """
    device = preds.device
    preds = preds.view(-1)
    targets = targets.view(-1)

    if num_classes is None:
        num_classes = int(max(preds.max().item(), targets.max().item()) + 1)

    f1_scores = []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue

        pred_c = preds == c
        targ_c = targets == c

        tp = (pred_c & targ_c).sum().float()
        fp = (pred_c & ~targ_c).sum().float()
        fn = (~pred_c & targ_c).sum().float()

        if tp == 0 and fp == 0 and fn == 0:
            # class không xuất hiện
            continue

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)

    if not f1_scores:
        return torch.tensor(0.0, device=device)
    return torch.stack(f1_scores).mean()
