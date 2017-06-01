from __future__ import absolute_import

from builtins import range
from operator import itemgetter
from future.utils import iteritems

from autograd.core import getval
import autograd.numpy as np
from autograd.container_types import make_dict, make_tuple, make_list
import autograd.util as au
from autograd_forward.convenience_wrappers import forward_derivative

COVERAGE_CHECKING = False

if COVERAGE_CHECKING:
    import coverage
    cover = coverage.Coverage()

def check_forward_grads(fun, *args):
    if COVERAGE_CHECKING:
        cover.load()
        cover.start()
    if not args:
        raise Exception("No args given")
    exact = tuple([forward_derivative(fun, i)(*args) for i in range(len(args))])
    args = [float(x) if isinstance(x, int) else x for x in args]
    numeric = au.nd(fun, *args)
    au.check_equivalent(exact, numeric)
    if COVERAGE_CHECKING:
        cover.stop()
        cover.save()

# This version of flatten should be merged into Autograd soon.
def flatten(value):
    """Flattens any nesting of tuples, arrays, or dicts.
       Returns 1D numpy array and an unflatten function.
       Doesn't preserve mixed numeric types (e.g. floats and ints).
       Assumes dict keys are sortable."""
    if isinstance(getval(value), np.ndarray):
        shape = value.shape
        def unflatten(vector):
            return np.reshape(vector, shape)
        return np.ravel(value), unflatten

    elif isinstance(getval(value), (float, int, complex)):
        return np.array([value]), lambda x : x[0]

    elif isinstance(getval(value), (tuple, list)):
        constructor = make_tuple if isinstance(getval(value), tuple) else make_list
        if not value:
            return np.array([]), lambda x : constructor()
        flat_pieces, unflatteners = zip(*map(flatten, value))
        split_indices = np.cumsum([len(vec) for vec in flat_pieces[:-1]])

        def unflatten(vector):
            pieces = np.split(vector, split_indices)
            return constructor(*[unflatten(v) for unflatten, v in zip(unflatteners, pieces)])

        return np.concatenate(flat_pieces), unflatten

    elif isinstance(getval(value), dict):
        items = sorted(iteritems(value), key=itemgetter(0))
        keys, flat_pieces, unflatteners = zip(*[(k,) + flatten(v) for k, v in items])
        split_indices = np.cumsum([len(vec) for vec in flat_pieces[:-1]])

        def unflatten(vector):
            pieces = np.split(vector, split_indices)
            return make_dict([(key, unflattener(piece))
                             for piece, unflattener, key in zip(pieces, unflatteners, keys)])

        return np.concatenate(flat_pieces), unflatten

    else:
        raise Exception("Don't know how to flatten type {}".format(type(value)))
