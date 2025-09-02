import numpy as np
from dataclasses import dataclass

@dataclass
class BinSpec:
    min_val: float
    max_val: float
    step: float

class Binner:
    def __init__(self, spec: BinSpec):
        edges = np.arange(spec.min_val, spec.max_val + 1e-6, spec.step)
        self.mids = (edges[:-1] + edges[1:]) / 2.0
        self.edges = edges
    def to_class(self, mos: float) -> int:
        # nearest midpoint
        idx = int(np.argmin(np.abs(self.mids - mos)))
        return idx
    def to_mos(self, cls_idx: int) -> float:
        return float(self.mids[cls_idx])
    def num_classes(self) -> int:
        return len(self.mids)
