import json
from typing import Type

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario

from .problem_instance import GNBG
from .solver import CMAESSolver, HMSDESolver, HMSSolver, ILSHADESolver, Solver


class Tuner:
    def __init__(
        self,
        n_trials: int | None = 100,
        max_n_evals: int | None = 10000,
        n_workers: int | None = 10,
    ):
        self.n_trials = n_trials
        self.max_n_evals = max_n_evals
        self.n_workers = n_workers

    def __call__(self, solver_class: Type[Solver]) -> dict:
        scenario = Scenario(
            solver_class().configspace,
            deterministic=False,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
        )
        experiment = HyperparameterOptimizationFacade(
            scenario,
            lambda config, seed: self.evaluate_solver(solver_class, config, seed),
        )
        best_found_config = experiment.optimize()
        return best_found_config.get_dictionary()

    def evaluate_solver(self, solver_class: Type[Solver], config: dict, seed: int) -> np.ndarray:
        fitness_values = []
        for fid in range(1, 25):
            problem = GNBG.read_from_file(fid)
            try:
                solver = solver_class(dict(config))
                solution = solver(problem, self.max_n_evals, seed)
                fitness_values.append(solution.fitness - problem.OptimumValue)
            except Exception as exc:
                print(f"{solver_class.__name__} failed, {exc}")
        return np.mean(np.array(fitness_values))


if __name__ == "__main__":
    solver_classes: list[Type[Solver]] = [
        HMSDESolver,
        HMSSolver,
        ILSHADESolver,
        CMAESSolver,
    ]
    for solver_class in solver_classes:
        tuner = Tuner(max_n_evals=10000, n_trials=1000)
        best_config = tuner(solver_class)
        with open(f"config/{solver_class().__class__.__name__}.json", "w") as json_file:
            json.dump(best_config, json_file)
