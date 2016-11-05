from defcon import *
from bratu import BratuProblem
import json

def test_bratu():
    problem = BratuProblem()
    dc = DeflatedContinuation(problem, teamsize=1, clear_output=True)
    dc.run(values={"lambda": list(arange(0.0, 3.6, 0.01)) + [3.6]})

    ac = ArclengthContinuation(problem, teamsize=1)
    ac.run(params=(0.5,), free="lambda", ds=0.1, sign=+1, bounds=(0.01, 3.6), branchids=[0])

    io = problem.io()
    io.setup(problem.parameters(), problem.functionals(), None)

    import backend
    if backend.comm_world.rank == 0:
        assert len(io.known_branches((3.0,))) == 2

        data = json.load(open("output/arclength/params-lambda=5.000000000000000e-01-freeindex-0-branchid-0-ds-1.00000000000000e-01.json", "r"))
        x = [entry[0] for entry in data]
        y = [entry[1][0] for entry in data]
        assert len(x) == len(y)
        assert len(x) > 0
