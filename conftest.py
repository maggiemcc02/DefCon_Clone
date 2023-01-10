import os
import functools
from petsc4py import PETSc
import pytest

starting_directory = None
def setup_module(module):
    global starting_directory
    starting_directory = os.getcwd()
    test_directory = os.path.dirname(str(module.fspath))
    os.chdir(test_directory)

def teardown_module(module):
    os.chdir(starting_directory)

def setup_function(function):
    """ setup any state tied to the execution of the given function.
    Invoked for every test function in the module.
    """
    opts = PETSc.Options()
    # should just do
    # opts.clear()
    # but it doesn't work!
    for key in opts.getAll():
        opts.delValue(key)

