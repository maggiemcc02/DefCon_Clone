# -*- coding: utf-8 -*-
import sys
from   math import floor

from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc
from defcon import *

import matplotlib.pyplot as plt

class ElasticaMooreSpenceProblem(BifurcationProblem):
    def mesh(self, comm):
        self.mycomm = comm
        return IntervalMesh(1000, 0, 1, comm=comm)

    def function_space(self, mesh):
        V = FunctionSpace(mesh, "CG", 1)
        R = FunctionSpace(mesh, "R", 0)
        return MixedFunctionSpace([V, R, V])

    def parameters(self):
        mu    = Constant(0)

        return [(mu, "mu", r"$\mu$")]

    def residual(self, z, params, w):
        mu = params[0]
        theta, lmbda, phi = split(z)
        ttheta, tlmbda, tphi = split(w)

        # Original PDE residual
        def pde_resid(theta, lmbda, ttheta):
            Fresid = (
                inner(grad(theta), grad(ttheta))*dx
                - lmbda**2*sin(theta)*ttheta*dx
                + mu*cos(theta)*ttheta*dx
            )
            return Fresid

        # Moore-Spence system
        F1 = pde_resid(theta, lmbda, ttheta)
        F2 = derivative(pde_resid(theta, lmbda, tphi), z, as_vector([phi, 0, 0]))
        F3 = inner(dot(phi, phi) - 1, tlmbda)*dx

        F = F1 + F2 + F3
        return F

    def boundary_conditions(self, Z, params):
        return [DirichletBC(Z.sub(0), 0.0, "on_boundary"), DirichletBC(Z.sub(2), 0.0, "on_boundary")]

    def functionals(self):
        def lambda_bif(z, params):
            with z.sub(1).dat.vec_ro as x:
                myparam = x.norm()
            return myparam
            
        def signedL2(z, params):
            # Argh.
            (theta, lmbda, phi) = z.split()
            j = sqrt(assemble(inner(theta, theta)*dx))
            g = project(grad(theta)[0], theta.function_space())
            return j*g((0.0,))

        return [(signedL2, "signedL2", r"$\theta'(0) \|\theta\|$"), (lambda_bif, "lambda_bif", r"$\lambda_\text{bifurcation}$")]
        

    def number_initial_guesses(self, params):
        return 1

    def initial_guess(self, Z, params, n):
        mu = params[0]
        x = SpatialCoordinate(Z.mesh())
        z = Function(Z)
        theta, lmbda, phi = z.split()
        # Now do initial solve
        msh = Z.mesh()
        V = FunctionSpace(msh, "CG", 1)
        th = Function(V)
        tth = TestFunction(V)
        lm = Constant(3.142)
        mu_ig = Constant(mu)
        def ig_resid(th, lm, tth):
            Figresid = (
                inner(grad(th), grad(tth))*dx
                - lm**2*inner(sin(th), tth)*dx
                + mu_ig*inner(cos(th), tth)*dx
            )
            return Figresid
        # Using guess for parameter lm, solve for state theta (th)
        A = ig_resid(th, lm, tth)
        bcs = [DirichletBC(V, 0.0, "on_boundary")]
        solve(A == 0, th, bcs=bcs)

        # Now solve eigenvalue problem for $F_u(u, \lambda)\phi = r\phi$
        # Want eigenmode phi with minimal eigenvalue r
        B = derivative(ig_resid(th, lm, TestFunction(V)), th, TrialFunction(V))

        petsc_M = assemble(inner(TrialFunction(V), TestFunction(V))*dx, bcs=bcs).petscmat
        petsc_B = assemble(B, bcs=bcs).petscmat

        num_eigenvalues = 1

        opts = PETSc.Options()
        opts.setValue("eps_target_magnitude", None)
        opts.setValue("eps_target", 0)
        opts.setValue("st_type", "sinvert")

        mycomm = self.mycomm
        es = SLEPc.EPS().create(comm=mycomm)
        es.setDimensions(num_eigenvalues)
        es.setOperators(petsc_B, petsc_M)
        es.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        es.setFromOptions()
        es.solve()

        ev_re, ev_im = petsc_B.getVecs()
        es.getEigenpair(0, ev_re, ev_im)
        eigenmode = Function(V)
        eigenmode.vector().set_local(ev_re)

        theta.assign(th)
        lmbda.assign(lm)
        phi.assign(eigenmode)
        return z

    def number_solutions(self, params):
        return 1 # Search only for the first bifurcation point

    def squared_norm(self, a, b, params):
        (theta, lmbda, phi) = split(a)
        (theta2, lmbda2, phi2) = split(b)
        return inner(theta - theta2, theta - theta2)*dx + inner(grad(theta - theta2), grad(theta - theta2))*dx + inner(lmbda - lmbda2, lmbda - lmbda2)*dx

    def save_pvd(self, z, pvd, params):
        (theta, lmbda, phi) = z.split()
        theta.rename("theta", "theta")
        phi.rename("phi", "phi")
        pvd.write(theta, phi)

    def solver_parameters(self, params, task, **kwargs):
        return {
            "snes_max_it": 100,
            "snes_atol": 1.0e-8,
            "snes_rtol": 0.0,
            "mat_type": "matfree",
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_converged_reason": None,
            "snes_linesearch_type": "basic",
            "ksp_type": "fgmres",
            "ksp_monitor_true_residual": None,
            "ksp_max_it": 10,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0,2",
            "pc_fieldsplit_1_fields": "1",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "python",
            "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
            "fieldsplit_0_assembled_pc_type": "lu",
            "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
            "mat_mumps_icntl_14": 200,
            "fieldsplit_1_ksp_type": "gmres",
            "fieldsplit_1_ksp_monitor_true_residual": None,
            "fieldsplit_1_ksp_max_it": 1,
            "fieldsplit_1_ksp_convergence_test": "skip",
            "fieldsplit_1_pc_type": "none",
         }

if __name__ == "__main__":
    dc = DeflatedContinuation(problem=ElasticaMooreSpenceProblem(), teamsize=1, verbose=True)
    dc.run(values={"mu": linspace(1.0, 0.0, 21)})

    dc.bifurcation_diagram("lambda_bif")
    plt.title(r"First bifurcation point for buckling of an Euler elastica as $\mu$ varies")
    plt.savefig("bifurcation_moore_spence.pdf")

