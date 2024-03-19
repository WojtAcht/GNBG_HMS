import random

import numpy as np
from cma import CMAEvolutionStrategy

from ..problem_instance import GNBG
from .solver import Solution, Solver


class CMAESSolver(Solver):
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
        sigma = 1.0
        cma = CMAEvolutionStrategy(
            x0,
            sigma,
            inopts={
                "bounds": [problem.MinCoordinate, problem.MaxCoordinate],
                "verbose": -9,
                "seed": random_state,
                "maxfevals": max_n_evals,
            },
        )
        cma.optimize(problem)
        return Solution(cma.result.xbest, cma.result.fbest, problem)
