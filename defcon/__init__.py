from mpi4py import MPI

try:
    import matplotlib
    matplotlib.use('PDF')
except ImportError:
    pass

import dolfin
assert dolfin.has_petsc4py()
dolfin.set_log_level(dolfin.ERROR)

from numpy                import arange, linspace
from bifurcationproblem   import BifurcationProblem
from defcon               import DeflatedContinuation
from iomodule             import IO, FileIO
