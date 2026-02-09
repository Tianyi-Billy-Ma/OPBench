import torch
import torch.nn.functional as F
from typing import Literal

from .functional import accuracy, f1_micro, f1_macro, auc_score


class Evaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def evaluate(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prefix: str = "",
    ) -> dict[str, float]:
        if logits.numel() == 0 or targets.numel() == 0:
            return self.empty_metrics(prefix)

        pred = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)

        metrics = {
            f"{prefix}acc": accuracy(pred, targets),
            f"{prefix}f1_micro": f1_micro(pred, targets),
            f"{prefix}f1_macro": f1_macro(pred, targets),
            f"{prefix}auc": auc_score(probs, targets, self.num_classes),
        }
        return metrics

    def empty_metrics(self, prefix: str) -> dict[str, float]:
        return {
            f"{prefix}acc": 0.0,
            f"{prefix}f1_micro": 0.0,
            f"{prefix}f1_macro": 0.0,
            f"{prefix}auc": 0.0,
        }
