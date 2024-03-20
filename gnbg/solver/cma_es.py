import random
from typing import TypedDict

import numpy as np
from cma import fmin

from ..problem_instance import GNBG
from .solver import Solution, Solver

CONFIG_FIELDS_OPTIONS = ["popsize", "tolfun"]


class CMAESConfig(TypedDict, total=False):
    popsize: int
    tolfun: float
    sigma0: float
    incpopsize: int
    restarts: int
    restart_from_best: bool


DEFAULT_CMAES_CONFIG = {
    "sigma0": 1.0,
    "incpopsize": 1,
    "restarts": 10,
    "restart_from_best": True,
}


class CMAESSolver(Solver):
    def __init__(self, config: CMAESConfig = {}):
        # Prepopulate the config with default values:
        self.config = DEFAULT_CMAES_CONFIG | config

    def __call__(
        self,
        problem: GNBG,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)
        x0 = np.random.uniform(problem.MinCoordinate, problem.MaxCoordinate, problem.Dimension)
        options = {
            "bounds": [problem.MinCoordinate, problem.MaxCoordinate],
            "verbose": -9,
            "maxfevals": max_n_evals,
            "seed": random_state,
        }
        options_from_config = {key: value for key, value in self.config.items() if key in CONFIG_FIELDS_OPTIONS}
        options |= options_from_config
        res = fmin(
            problem,
            x0,
            sigma0=self.config["sigma0"],
            options=options,
            bipop=True,
            restart_from_best=self.config["restart_from_best"],
            restarts=self.config["restarts"],
            incpopsize=self.config["incpopsize"],
        )
        return Solution(res[0], res[1], problem)
