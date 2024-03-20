from time import time

import numpy as np
import pandas as pd
from multiprocess import Pool

from .problem_instance import GNBG
from .solver import CMAESSolver, HMSSolver, ILSHADESolver, Solver


class GNBGProblemEvaluator:
    def __init__(
        self,
        solvers: list[Solver],
        seed_count: int = 30,
        max_n_evals: int = 10000,
        max_pool: int = 10,
    ):
        self.seed_count = seed_count
        self.solvers = solvers
        self.max_n_evals = max_n_evals
        self.max_pool = max_pool

    def evaluate_solver(self, solver: Solver, problem: GNBG, max_n_evals: int) -> np.ndarray:
        fitness_values = []
        for random_state in range(1, self.seed_count + 1):
            try:
                solution = solver(problem, max_n_evals, random_state)
                fitness_values.append(solution.fitness - problem.OptimumValue)
            except Exception as exc:
                print(f"{solver.__class__.__name__} failed, {exc}")
        return np.array(fitness_values)

    def evaluate_problem(self, problem: GNBG) -> pd.DataFrame:
        start = time()
        rows = []
        for solver in self.solvers:
            fitness_values = self.evaluate_solver(solver, problem, self.max_n_evals)
            rows.append(
                {
                    "problem_id": problem.FID,
                    "solver": solver.__class__.__name__,
                    "fitness_mean": np.mean(fitness_values),
                    "fitness_std": np.std(fitness_values),
                }
            )
        end = time()
        print(f"Problem {problem.FID} evaluated in {(end - start):.2f} seconds")
        return pd.DataFrame(rows)

    def __call__(self) -> pd.DataFrame:
        with Pool(self.max_pool) as p:
            problems = [GNBG.read_from_file(fid) for fid in range(1, 25)]
            pool_outputs = p.map(lambda problem: self.evaluate_problem(problem), problems)
        return pd.concat(pool_outputs)


solvers = [ILSHADESolver(), CMAESSolver(), HMSSolver()]

evaluator = GNBGProblemEvaluator(solvers, seed_count=10)
results = evaluator()
results.to_csv("results.csv")
