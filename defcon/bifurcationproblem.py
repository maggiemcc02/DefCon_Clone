# -*- coding: utf-8 -*-
import backend
import iomodule
import nonlinearproblem, nonlinearsolver

class BifurcationProblem(object):
    """
    A base class for bifurcation problems.

    This object is overridden by the user to implement his particular problem.
    """

    def mesh(self, comm):
        """
        This method loads/builds the mesh with a given communicator.

        *Arguments*
          comm (MPI communicator)
            The MPI communicator to use in building the mesh.

            Typically, the MPI communicator passed in here will be shared among
            a small number of the processors (called a team). Each team solves a
            PDE independently.
        *Returns*
          mesh (:py:class:`dolfin.Mesh`)
        """
        raise NotImplementedError

    def function_space(self, mesh):
        """
        This method creates the function space for the prognostic variables of
        the problem.

        *Arguments*
          mesh (:py:class:`dolfin.Mesh`)
            The mesh to use.
        *Returns*
          functionspace (:py:class:`dolfin.FunctionSpace`)
        """
        raise NotImplementedError

    def parameters(self):
        """
        This method returns a list of tuples. Each tuple contains (Constant,
        asciiname, tex). For example, if there is one parameter

        lmbda = Constant(...)

        to be varied, this routine should return

        [(lmbda, "lambda", r"$\lambda$")]

        You may need to add

        # -*- coding: utf-8 -*-

        to the top of the Python script to use UTF characters.

        The values in the Constants are irrelevant; they are initialised in the
        continuation.

        *Returns*
          params
            A list of [(Constant, asciiname, tex), ...]
        """
        raise NotImplementedError

    def residual(self, state, params, test):
        """
        This method defines the PDE to be solved: if you would solve the PDE via

        solve(F == 0, state, ...)

        then this method should return F.

        The parameters will be varied internally by the continuation algorithm,
        i,e.  this will not be called multiple times for multiple parameters.

        *Arguments*
          state (:py:class:`dolfin.Function`)
            a Function in the FunctionSpace
          params (tuple of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
          test  (:py:class:`dolfin.TestFunction`)
            the test function to use in defining the residual
        *Returns*
          residual (:py:class:`ufl.form.Form`)
        """
        raise NotImplementedError

    def boundary_conditions(self, function_space, params):
        """
        This method returns a list of DirichletBC objects to impose on the
        problem.

        *Arguments*
          function_space (:py:class:`dolfin.FunctionSpace`)
            the function space returned by self.function_space()
          params (list of :py:class:`dolfin.Constant`)
            the parameters to use, in the same order returned by parameters()
        *Returns*
          bcs (list of :py:class:`dolfin.DirichletBC`)
        """
        raise NotImplementedError

    def functionals(self):
        """
        This method returns a list of functionals. Each functional is a tuple
        consisting of a callable, an ascii name, and a tex label.  The callable
        J is called via

          j = J(state, params)

        and should return a float.

        For example, this routine might consist of

        def functionals(self):
            def L2norm(state, param):
                return assemble(inner(state, state)*dx)**0.5
            return [(L2norm, "L2norm", r"\|y\|")]

        *Returns*
          functionals (list of tuples)
        """
        raise NotImplementedError

    def number_initial_guesses(self, params):
        """
        Return the number of initial guesses we wish to search from at the
        very first deflation step, i.e. when initialising the continuation.
        """
        raise NotImplementedError

    def initial_guess(self, function_space, params, n):
        """
        Return the n^{th} initial guess.
        """
        raise NotImplementedError

    def number_solutions(self, params):
        """
        If the number of solutions for a given set of parameters is analytically
        known, then this function can return it. In this case the continuation
        algorithm will stop looking when it has found that many solutions.
        Otherwise the routine should return float("inf").

        *Arguments*
          params (tuple of :py:class:`float`)
            parameters to use, in the same order returned by parameters()

        *Returns*
          nsolutions (int)
            number of solutions, known from analysis, or float("inf")
        """
        return float("inf")

    def squared_norm(self, state1, state2, params):
        """
        This method computes the squared-norm between two vectors in the Hilbert
        space defined in the function_space method.

        This is used to define the norm used in deflation, and to test two
        functions for equality, among other things.

        The default is
            def squared_norm(self, state1, state2, params)
                return inner(state1 - state2, state1 - state2)*dx

        *Arguments*
        """
        return backend.inner(state1 - state2, state1 - state2)*backend.dx

    def trivial_solutions(self, function_space, params, freeindex):
        """
        This method returns any trivial solutions of the problem,
        i.e. solutions u such that f(u, \lambda) = 0 for all \lambda.
        These will be deflated at every computation.

        freeindex is the index into params that is being modified by this
        current continuation run. This is useful if the trivial solutions
        depend on the values of parameters in params other than freeindex.
        """
        return []

    def monitor(self, params, branchid, solution, functionals):
        """
        This method is called whenever a solution is computed.
        The user can specify custom processing tasks here.
        """
        pass

    def io(self, prefix=""):
        """
        Return an IO object that defcon will use to save solutions and functionals.

        The default is usually a good choice.
        """

        return iomodule.SolutionIO(prefix + "output")

    def save_pvd(self, y, pvd):
        """
        Save the function y to a PVD file.

        The default is

        pvd << y
        """
        pvd << y

    def assembler(self, F, y, bcs):
        """
        The class used to assemble the nonlinear problem.

        Most users will never need to override this: it's only useful if you
        want to do something unusual in the assembly process.
        """
        if backend.__name__ == "dolfin":
            return nonlinearproblem.GeneralProblem(F, y, bcs)
        else:
            return backend.NonlinearVariationalProblem(F, y, bcs)

    def solver(self, problem, prefix=""):
        """
        The class used to solve the nonlinear problem.

        Users might want to override this if they want to customize
        how the nonlinear solver is set up.
        """
        if backend.__name__ == "dolfin":
            return nonlinearsolver.SNUFLSolver(
                problem, prefix=prefix, **self.solver_kwargs(problem)
            )
        else:
            return backend.NonlinearVariationalSolver(
                problem, options_prefix=prefix,
                **self.solver_kwargs(problem)
            )

    def solver_kwargs(self, problem):
        """
        This allows users to override to pass keyword arguments into either
        the DOLFIN or Firedrake backend solvers.  For example, Firedrake
        allows the user to set nullspaces for matrices or special
        context for preconditioning matrix-free matrices.

        Note: we use the 'problem' to instantiate this since it's
        known to the solver and it already has particular meshes
        and function spaces instantiated in it.
        """
        return {}
        
