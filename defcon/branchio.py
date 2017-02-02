from __future__ import absolute_import

from defcon import iomodule
import defcon.backend as backend
from defcon.backend import HDF5File, Function

import h5py # FIXME: remove dependency on h5py, eventually
from numpy import array

import cPickle as pickle
import os
import glob
import collections
import sys
import time
import traceback
from ast import literal_eval

# If you're storing one branch per hdf5 file, the
# hardest part is figuring out which parameters have which
# branches. I'm going to use a special HDF5 file for this
# purpose.

def paramstokey(params): return "(" + ", ".join("%.15e" % x for x in params) + ")"
def keytoparams(key):
    out = literal_eval(key)
    if isinstance(out, float): return (out,)
    else:                      return out

class ParameterMap(object):
    def __init__(self, directory):
        self.path = os.path.join(directory, "parameter_map.pck")
        try:
            f = open(self.path, "r")
            self.dict = pickle.load(f)
        except:
            self.dict = collections.defaultdict(list)

    def write(self):
        with open(self.path, "w") as f:
            pickle.dump(self.dict, f)

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, key, value):
        self.dict[key] = value
        self.write()

class BranchIO(iomodule.SolutionIO):
    def __init__(self, directory):
        iomodule.SolutionIO.__init__(self, directory)
        self.pm = None

    def parameter_map(self):
        if self.pm is None:
            self.pm = ParameterMap(self.directory)
        return self.pm

    def close_parameter_map(self):
        self.pm.write()
        self.pm = None

    def known_branches(self, params):
        if self.worldcomm.rank != 0:
            raise ValueError("Can only be requested on master process")

        if self.pm is None:
            self.pm = ParameterMap(self.directory)
        known = self.pm[params]
        return known

    def solution_filename(self, branchid):
        return os.path.join(self.directory, "solution-" + str(branchid) + ".h5")

    def temp_solution_filename(self, branchid):
        return os.path.join(self.tmpdir, "solution-" + str(branchid) + ".h5")

    def stability_filename(self, branchid):
        return os.path.join(self.directory, "stability-" + str(branchid) + ".h5")

    def save_solution(self, solution, funcs, params, branchid):
        key = paramstokey(params)
        fname = self.solution_filename(branchid)
        tmpname = self.temp_solution_filename(branchid)

        # I can't believe I have to do this to get filesystem synchronisation right.
        if self.pcomm.rank == 0:
            if os.path.exists(fname):
                mode = "a"
                os.rename(fname, tmpname)
            else:
                mode = "w"
            self.mcomm.bcast(mode)
        else:
            mode = self.mcomm.bcast(None)

        with HDF5File(comm=self.pcomm, filename=tmpname, file_mode=mode) as f:
            if self.pcomm.size > 1:
                f.set_mpi_atomicity(True)
            f.write(solution, key + "/solution")

            attrs = f.attributes(key)
            for (func, value) in zip(self.functionals, funcs):
                attrs[func[1]] = value
        self.pcomm.Barrier()

        # Trick from Lawrence Mitchell: POSIX guarantees that mv is atomic
        if self.pcomm.rank == 0:
            os.rename(tmpname, fname)

            # Belt-and-braces: sleep until the path exists
            while not os.path.exists(fname):
                time.sleep(0.1)
        self.pcomm.Barrier()

        return

    def fetch_solutions(self, params, branchids):
        key = paramstokey(params)
        solns = []
        for branchid in branchids:
            failcount = 0
            filename = self.solution_filename(branchid)

            while True:
                try:
                    with HDF5File(comm=self.pcomm, filename=filename, file_mode="r") as f:
                        solution = Function(self.function_space)
                        f.read(solution, key + "/solution")
                        solns.append(solution)
                        break
                except Exception:
                    failcount += 1
                    if failcount % 100 == 0:
                        self.log("Could not load %s from file %s; backtrace and exception follows" % (key, filename), warning=True)
                        traceback.print_stack()
                        traceback.print_exc()
                    if failcount == 100000:
                        self.log("Failed a great many times. Raising exception.", warning=True)
                        raise
                    time.sleep(0.1)

        return solns

    def fetch_functionals(self, params, branchid):
        funcs = []
        filename = self.solution_filename(branchid)
        if not os.path.exists(filename):
            return funcs

        with HDF5File(comm=self.pcomm, filename=self.solution_filename(branchid), file_mode="r") as f:
            for param in params:
                key = paramstokey(param)

                attrs = f.attributes(key)

                thesefuncs = []
                for func in self.functionals:
                    thesefuncs.append(attrs[func[1]])
                funcs.append(thesefuncs)

        return funcs

    def known_parameters(self, fixed, branchid):
        out = []
        filename = self.solution_filename(branchid)
        if not os.path.exists(filename):
            return out

        fixedindices = []
        fixedvalues  = []
        n = len(fixed)

        for (i, param) in enumerate(self.parameters):
            if param[1] in fixed:
                fixedindices.append(i)
                fixedvalues.append(fixed[param[1]])

        with h5py.File(filename, "r") as f:
            for key in f['/']:
                params = keytoparams(key)
                add = True

                for i in range(n):
                    if params[fixedindices[i]] != fixedvalues[i]:
                        add = False
                        break
                if add:
                    out.append(params)

        return out

    def max_branch(self):
        filenames = glob.glob(os.path.join(self.directory, 'solution*.h5'))
        maxbranch = max(int(filename.split('-')[-1][:-3]) for filename in filenames)
        return maxbranch

    def pickle(self, obj):
        return pickle.dumps(obj, protocol=0)

    def unpickle(self, s):
        return pickle.loads(s)

    def save_stability(self, stable, eigenvalues, eigenfunctions, params, branchid):
        assert len(eigenvalues) == len(eigenfunctions)

        key = paramstokey(params)
        fname = self.stability_filename(branchid)
        if os.path.exists(fname):
            mode = "a"
            exists = True
        else:
            mode = "w"
            exists = False

        with HDF5File(comm=self.pcomm, filename=fname, file_mode=mode) as f:
            if self.pcomm.size > 1:
                f.set_mpi_atomicity(True)

            # dummy dataset so that we can actually store the attribute
            if backend.__name__ == "firedrake":
                f._h5file[key + "/stability"] = array([0.0])
            else:
                f.write(array([0.0]), key + "/stability")

            attrs = f.attributes(key + "/stability")
            attrs["stable"] = self.pickle(stable)

            if len(eigenvalues) > 0:
                for (j, (eigval, eigfun)) in enumerate(zip(eigenvalues, eigenfunctions)):
                    ekey = key + "/stability/eigenfunction-%d" % j
                    f.write(eigfun, ekey)
                    f.attributes(ekey)["eigenvalue"] = eigval
        self.pcomm.barrier()

    def fetch_stability(self, params, branchids):
        stables = []
        key = paramstokey(params)

        for branchid in branchids:
            fname = self.stability_filename(branchid)
            with HDF5File(comm=self.pcomm, filename=fname, file_mode="r") as f:
                attrs = f.attributes(key + "/stability")
                if "stable" not in attrs:
                    msg = "Could not find stability information for %s in %s." % (params, fname)
                    raise KeyError(msg)
                stables.append(self.unpickle(attrs["stable"]))

        return stables
