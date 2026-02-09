import torch
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0:
        return 0.0
    correct = (pred == target).sum().item()
    return correct / pred.numel()


def f1_micro(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0:
        return 0.0
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    return float(f1_score(target_np, pred_np, average="micro", zero_division="warn"))


def f1_macro(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0:
        return 0.0
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    return float(f1_score(target_np, pred_np, average="macro", zero_division="warn"))


def auc_score(probs: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    if probs.numel() == 0:
        return 0.0

    probs_np = probs.cpu().numpy()
    target_np = target.cpu().numpy()

    if num_classes == 2:
        if probs_np.ndim == 2:
            probs_np = probs_np[:, 1]
        try:
            return float(roc_auc_score(target_np, probs_np))
        except ValueError:
            return 0.0
    else:
        try:
            return float(
                roc_auc_score(target_np, probs_np, multi_class="ovr", average="macro")
            )
        except ValueError:
            return 0.0
