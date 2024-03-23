from .evaluator import GNBGProblemEvaluator
from .solver import CMAESSolver, HMSDESolver, HMSSolver, ILSHADESolver

solvers = [
    ILSHADESolver.from_config(),
    CMAESSolver.from_config(),
    HMSSolver.from_config(),
    HMSDESolver.from_config(),
]

evaluator = GNBGProblemEvaluator(solvers, seed_count=10)
results = evaluator()
results.to_csv("results.csv")
