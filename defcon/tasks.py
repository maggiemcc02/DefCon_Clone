from __future__ import absolute_import

import functools

@functools.total_ordering
class Task(object):
    """
    A base class for Tasks.
    """
    def __le__(self, other):
        """Comparison operator needed in heapq"""
        return id(self) < id(other)

class QuitTask(Task):
    """
    A task indicating the worker should quit.
    """
    def __str__(self):
        return "QuitTask"

class ContinuationTask(Task):
    """
    A base task for continuing a known branch.

    *Arguments*
      taskid (int)
        Global identifier for this task
      oldparams (tuple)
        Parameter values to continue from
      freeindex (int)
        Which parameter is currently being varied
      branchid (int)
        Which branch to continue (int)
      newparams (tuple)
        Parameter values to continue to
      direction (+1 or -1)
        +1 means go increasing in parameter direction; -1 to go backwards
    """
    def __init__(self, taskid, oldparams, freeindex, branchid, newparams, direction, ensure_branches=None):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.freeindex = freeindex
        self.branchid  = branchid
        self.newparams = newparams
        self.direction = direction
        assert isinstance(branchid, int)
        assert self.oldparams != self.newparams

        if ensure_branches is None: ensure_branches = set()
        self.ensure_branches = ensure_branches

    def ensure(self, branches):
        self.ensure_branches.update(branches)

    def __str__(self):
        return "ContinuationTask(taskid=%s, oldparams=%s, freeindex=%s, branchid=%s, newparams=%s, direction=%s)" % (self.taskid, self.oldparams, self.freeindex, self.branchid, self.newparams, self.direction)

class DeflationTask(Task):
    """
    A task that seeks new, unknown solutions for a given parameter
    value.

    *Arguments*
      taskid (int)
        Global identifier for this task
      oldparams (tuple)
        Parameter values to continue from. If None, this means use the initial guesses
      freeindex (int)
        Which parameter is currently being varied
      branchid (int)
        Which branch to search from (int). If oldparams is None, this is the number of the initial guess
        to use
      newparams (tuple)
        Parameter values to continue to
      ensure_branches (set):
        Branches that *must* be deflated; if they are not present yet, wait for them
    """
    def __init__(self, taskid, oldparams, freeindex, branchid, newparams, ensure_branches=None):
        self.taskid    = taskid
        self.oldparams = oldparams
        self.freeindex = freeindex
        self.branchid  = branchid
        self.newparams = newparams
        assert isinstance(branchid, int)
        assert self.oldparams != self.newparams

        if ensure_branches is None: ensure_branches = set()
        self.ensure_branches = ensure_branches

    def ensure(self, branches):
        self.ensure_branches.update(branches)

    def __str__(self):
        return "DeflationTask(taskid=%s, oldparams=%s, freeindex=%s, branchid=%s, newparams=%s)" % (self.taskid, self.oldparams, self.freeindex, self.branchid, self.newparams)

class StabilityTask(Task):
    """
    A task that computes the stability of solutions we have found.

    *Arguments*
      taskid (int)
        Global identifier for this task
      oldparams (tuple)
        Parameter values to investigate.
      freeindex (int)
        Which parameter is currently being varied
      branchid (int)
        Which branch to investigate.
      direction (+1 or -1)
        Whether to go forwards or backwards
      hint (anything)
        A hint to pass to the stability calculation.
    """
    def __init__(self, taskid, oldparams, freeindex, branchid, direction, hint):
        self.taskid = taskid
        self.oldparams = oldparams
        self.freeindex = freeindex
        self.branchid = branchid
        self.direction = direction
        self.hint = hint
        self.extent = None

    def __str__(self):
        return "StabilityTask(taskid=%s, params=%s, freeindex=%s, branchid=%s, direction=%s)" % (self.taskid, self.oldparams, self.freeindex, self.branchid, self.direction)

    def set_extent(self, extent):
        # Set the known branch extent: this is used by the task to decide whether to keep
        # going or not
        if self.extent is not None:
            assert False # only allow it to be set once
        self.extent = list(extent)

class ArclengthTask(Task):
    """
    A task that computes the arclength continuation of a solution.

    *Arguments*
      taskid (int)
        Global identifier for this task
      params (tuple)
        Parameter values to start from
      branchid (int)
        Which branch to continue
      bounds (tuple of floats)
        Upper and lower bounds of interest
      sign (+1 or -1)
        Whether to initially continue forwards or backwards in parameter
      ds (float)
        Size of step in arclength
    """
    def __init__(self, taskid, params, branchid, bounds, sign, ds):
        self.taskid = taskid
        self.params = params
        self.branchid = branchid
        self.bounds = bounds
        self.sign = sign
        self.ds = ds

    def __str__(self):
        return "ArclengthTask(taskid=%s, params=%s, branchid=%s, sign=%s, ds=%s, bounds=%s)" % (self.taskid, self.params, self.branchid, self.sign, self.ds, self.bounds)

class Response(object):
    """
    A class that encapsulates whether a given task was successful or not."
    """
    def __init__(self, taskid, success, data=None):
        self.taskid = taskid
        self.success = success
        self.data = data

    def __str__(self):
        return "Response(taskid=%s, success=%s, data=%s)" % (self.taskid, self.success, self.data)
