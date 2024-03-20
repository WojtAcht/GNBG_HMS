import random
from typing import TypedDict

import numpy as np
import pyade.ilshade

from ..problem_instance import GNBG
from .solver import Solution, Solver


class ILSHADEConfig(TypedDict, total=False):
    population_size: int
    memory_size: int


class ILSHADESolver(Solver):
    def __init__(self, config: ILSHADEConfig | None = {}):
        self.config = config

    def __call__(
        self,
        problem: GNBG,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        random.seed(random_state)
        np.random.seed(random_state)
        algorithm = pyade.ilshade
        # pyade requires dim to be int:
        params = algorithm.get_default_params(dim=int(problem.Dimension))
        params["bounds"] = np.array(
            [[problem.MinCoordinate, problem.MaxCoordinate]] * problem.Dimension,
            dtype=float,
        )
        for key, value in self.config.items():
            params[key] = value
        params["func"] = problem
        params["max_evals"] = max_n_evals
        params["seed"] = random_state
        x, fitness = algorithm.apply(**params)
        return Solution(x, fitness, problem)
