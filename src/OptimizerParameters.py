from typing import *
from dataclasses import dataclass


@dataclass(frozen=True)
class AdamOptimizerParameters:
    lr: float
    betas: Tuple[float, float]
    weight_decay: float
