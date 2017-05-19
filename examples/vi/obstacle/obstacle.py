# -*- coding: utf-8 -*-
from defcon import *
from dolfin import *

class Obstacle(Expression):
    def eval(self, values, x):
        if x[0] < -0.5:
            values[0] = -0.2
            return
        if -0.5 <= x[0] <= 0.0:
            values[0] = -0.4
            return
        if 0.0 <= x[0] < 0.5:
            values[0] = -0.6
            return
        if 0.5 <= x[0] <= 1.0:
            values[0] = -0.8
            return
lb = Obstacle(degree=1)
ub = Constant(1e20)

class ObstacleProblem(BifurcationProblem):
    def mesh(self, comm):
        return RectangleMesh(comm, Point(-1, -1), Point(1, 1), 64, 64, "crossed")

    def coarse_meshes(self, comm):
        return [RectangleMesh(comm, Point(-1, -1), Point(1, 1), 16, 16, "crossed"), RectangleMesh(comm, Point(-1, -1), Point(1, 1), 32, 32, "crossed")]

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 1)
        print("V.dim(): %d" % V.dim())
        return V

    def parameters(self):
        f = Constant(0)
        return [(f, "f", r"$f$")]

    def residual(self, u, params, v):
        f = params[0]

        F = (
              inner(grad(u), grad(v))*dx
            - inner(f, v)*dx
            )

        return F

    def boundary_conditions(self, V, params):
        return [DirichletBC(V, 0, "on_boundary")]

    def functionals(self):
        def uL2(u, params):
            j = assemble(u*u*dx)
            return j

        return [(uL2, "uL2", r"$\|u\|^2$")]

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, V, params, n):
        return Function(V)

    def number_solutions(self, params):
        return 1

    def solver_parameters(self, params, klass):
        return {
               "snes_max_it": 50,
               "snes_atol": 1.0e-9,
               "snes_rtol": 1.0e-9,
               "snes_monitor": None,
               "ksp_type": "preonly",
               "ksp_monitor": None,
               "ksp_rtol": 1.0e-10,
               "ksp_atol": 1.0e-10,
               "pc_type": "lu",
               "pc_factor_mat_solver_package": "mumps",
               }

    def save_pvd(self, z, pvd):
        print("wtf")
        z.rename("Solution", "Solution")
        print("z.function_space().dim(): %d" % z.function_space().dim())
        pvd << z

if __name__ == "__main__":
    eqproblem = ObstacleProblem()
    viproblem = VIBifurcationProblem(eqproblem, lb, ub)

    dc = DeflatedContinuation(problem=viproblem, teamsize=1, verbose=True, clear_output=True)
    dc.run(values={"f": -10})
