# -*- coding: utf-8 -*-
import sys
from   math import floor

from defcon import *
from dolfin import *

import matplotlib.pyplot as plt

from petsc4py import PETSc

class ElasticaProblem(BifurcationProblem):
    def __init__(self):
        self.bcs = None
    def mesh(self, comm):
        return IntervalMesh(comm, 1000, 0, 1)

    def function_space(self, mesh):
        return FunctionSpace(mesh, "CG", 1)

    def parameters(self):
        lmbda = Constant(0)
        mu    = Constant(0)

        return [(lmbda, "lambda", r"$\lambda$"),
                (mu,    "mu",     r"$\mu$")]

    def residual(self, theta, params, v):
        (lmbda, mu) = params

        F = (
              inner(grad(theta), grad(v))*dx
              - lmbda**2*sin(theta)*v*dx
              + mu*v*dx
            )

        return F

    def boundary_conditions(self, V, params):
        if self.bcs is None:
            self.bcs = [DirichletBC(V, 0.0, "on_boundary")]
        return self.bcs

    def functionals(self):
        def signedL2(theta, params):
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g((0.0,))

        return [(signedL2, "signedL2", r"$\theta'(0) \|\theta\|$")]

    def trivial_solutions(self, V, params, freeindex):
        # check we're continuing in lambda:
        if freeindex == 0:
            # check if mu is 0
            if params[1] == 0.0:
                # return the trivial solution
                return [Function(V)]
        return []

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return interpolate(Constant(1), V)

    def number_solutions(self, params):
        # Here I know the number of solutions for each value of lambda.
        # This cheating allows me to calculate the bifurcation diagram
        # much more quickly. This can be disabled without changing the
        # correctness of the calculations.

        (lmbda, mu) = params
        n = int(floor((lmbda/pi)))*2 # this is the exact formula for mu = 0, but works for mu = 0.5 also

        if mu == 0: return max(n, 1) # don't want the trivial solution
        else:       return n + 1

    def squared_norm(self, a, b, params):
        return inner(a - b, a - b)*dx + inner(grad(a - b), grad(a - b))*dx

    def compute_stability(self, params, branchid, theta, hint=None):
        if params[0] == 0: return {"stable": True}

        V = theta.function_space()
        trial = TrialFunction(V)
        test  = TestFunction(V)

        bcs = self.boundary_conditions(V, params)
        comm = V.mesh().mpi_comm()

        F = self.residual(theta, map(Constant, params), test)
        J = derivative(F, theta, trial)
        b = inner(Constant(1), test)*dx # a dummy linear form, needed to construct the SystemAssembler

        # Build the LHS matrix
        A = PETScMatrix(comm)
        asm = SystemAssembler(J, b, bcs)
        asm.assemble(A)

        pc = PETSc.PC().create(comm)
        pc.setOperators(A.mat())
        pc.setType("cholesky")
        pc.setFactorSolverPackage("mumps")
        pc.setUp()

        F = pc.getFactorMatrix()
        (neg, zero, pos) = F.getInertia()

        print "Inertia: (-: %s, 0: %s, +: %s)" % (neg, zero, pos)

        expected_dim = 0

        # Nocedal & Wright, theorem 16.3
        if neg == expected_dim:
            is_stable = True
        else:
            is_stable = False

        d = {"stable": is_stable}
        return d

    def solver_parameters(self, params, klass):
        args = {
               "snes_max_it": 100,
               "snes_atol": 1.0e-9,
               "snes_rtol": 0.0,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "pc_type": "lu",
               "mat_mumps_icntl_24": 1,
               "mat_mumps_icntl_13": 1
               }
        return args

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ElasticaProblem(), teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"lambda": linspace(0, 3.9*pi, 200), "mu": [0.5]}, freeparam="lambda")

    dc.bifurcation_diagram("signedL2", fixed={"mu": 0.5})
    plt.title(r"Buckling of an Euler elastica, $\mu = 1/2$")
    plt.savefig("bifurcation.pdf")
