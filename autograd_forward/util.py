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
