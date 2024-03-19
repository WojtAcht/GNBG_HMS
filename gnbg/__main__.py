from .problem_instance import GNBG
from .solver.cma_es import CMAESSolver

for fid in range(1, 25):
    problem = GNBG.read_from_file(fid)
    solution = CMAESSolver()(problem, 10000, 1)
    print(f"Problem {fid}: {solution.fitness}")
    solution.problem.plot()
