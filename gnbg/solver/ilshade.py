import json
import random
from typing import TypedDict

import numpy as np
import pyade.ilshade
from ConfigSpace import ConfigurationSpace

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

    @property
    def configspace(self) -> ConfigurationSpace:
        return ConfigurationSpace(
            {
                "population_size": (10, 300),
                "memory_size": (1, 20),
            }
        )

    @classmethod
    def from_config(cls) -> "ILSHADESolver":
        with open(f"config/{cls.__name__}.json", "r") as json_file:
            config = json.load(json_file)
        return cls(config)
