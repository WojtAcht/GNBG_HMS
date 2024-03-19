from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..problem_instance import GNBG


@dataclass
class Solution:
    x: np.ndarray
    fitness: float
    problem: GNBG


class Solver(Protocol):
    def __call__(self, problem: GNBG, max_n_evals: int, random_state: int) -> Solution:
        ...
