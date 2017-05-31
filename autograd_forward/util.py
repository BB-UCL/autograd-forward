from __future__ import absolute_import

from builtins import range

import autograd.util as au
from autograd_forward.convenience_wrappers import forward_derivative

def check_forward_grads(fun, *args):
    if not args:
        raise Exception("No args given")
    exact = tuple([forward_derivative(fun, i)(*args) for i in range(len(args))])
    args = [float(x) if isinstance(x, int) else x for x in args]
    numeric = au.nd(fun, *args)
    au.check_equivalent(exact, numeric)

# This version of flatten should be merged into Autograd soon.
def flatten(value):
    """Flattens any nesting of tuples, arrays, or dicts.
       Returns 1D numpy array and an unflatten function.
       Doesn't preserve mixed numeric types (e.g. floats and ints).
       Assumes dict keys are sortable."""
    try:
        vs = vspace(getval(value))
    except TypeError:
        raise Exception("Don't know how to flatten type {}".format(type(value)))
    return vs.flatten(value), vs.unflatten
