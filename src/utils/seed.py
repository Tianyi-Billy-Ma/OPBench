import pytorch_lightning as pl


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    pl.seed_everything(seed)
