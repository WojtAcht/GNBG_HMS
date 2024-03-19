from .solver import Solver, Solution
from ..problem_instance import GNBG
from pyhms import (
    CMALevelConfig,
    DemeTree,
    EALevelConfig,
    EvalCutoffProblem,
    DontStop,
    SEA,
    SingularProblemEvalLimitReached,
    get_NBC_sprout,
    hms,
)
from leap_ec.problem import FunctionProblem
import numpy as np


class HMSSolver(Solver):
    def __call__(
        self,
        problem: GNBG,
        max_n_evals: int,
        random_state: int | None = None,
    ) -> Solution:
        function_problem = FunctionProblem(problem, maximize=False)
        problem_with_cutoff = EvalCutoffProblem(function_problem, max_n_evals)
        bounds = np.array(
            [[problem.MinCoordinate, problem.MaxCoordinate]] * problem.Dimension,
            dtype=float,
        )

        config = [
            EALevelConfig(
                ea_class=SEA,
                generations=2,
                problem=problem_with_cutoff,
                bounds=bounds,
                pop_size=20,
                mutation_std=10.0,
                lsc=DontStop(),
            ),
            CMALevelConfig(
                generations=5,
                problem=problem_with_cutoff,
                bounds=bounds,
                sigma0=1.0,
                lsc=DontStop(),
            ),
        ]
        global_stop_condition = SingularProblemEvalLimitReached(max_n_evals)
        sprout_condition = get_NBC_sprout(level_limit=4)
        hms_tree = hms(
            config,
            global_stop_condition,
            sprout_condition,
            {"random_seed": random_state},
        )
        return Solution(
            hms_tree.best_individual.genome, hms_tree.best_individual.fitness, problem
        )
