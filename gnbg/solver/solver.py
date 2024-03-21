from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ConfigSpace import ConfigurationSpace

from ..problem_instance import GNBG


@dataclass
class Solution:
    x: np.ndarray
    fitness: float
    problem: GNBG


class Solver(Protocol):
    def __init__(self, config: dict | None = None):
        ...

    def __call__(self, problem: GNBG, max_n_evals: int, random_state: int) -> Solution:
        ...

    @property
    def configspace(self) -> ConfigurationSpace:
        ...

    @classmethod
    def from_config(cls):
        ...
