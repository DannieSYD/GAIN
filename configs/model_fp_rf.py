from dataclasses import dataclass, field


@dataclass
class RandomForest:
    n_jobs: int = 8
    n_estimators: int = 500
    min_samples_leaf: int = 2
    min_samples_split: int = 10
    min_impurity_decrease: int = 0
    warm_start: bool = True


@dataclass
class E3FP:
    bits: int = 4096
    radius_multiplier: float = 1.5
    rdkit_invariants: bool = True


@dataclass
class ModelFPRF:
    modality: str = '3D'
    max_workers: int = 24
    e3fp: E3FP = field(default_factory=E3FP)
    random_forest: RandomForest = field(default_factory=RandomForest)
