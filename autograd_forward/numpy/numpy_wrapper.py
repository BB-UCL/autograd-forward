from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy.numpy_wrapper as nw
from autograd.core import isnode, vspace


def array_from_args_fwd_gradmaker(argnum, g, ans, gvs, vs, args, kwargs):
    result = list()
    for i, arg in enumerate(args):
        if i == argnum:
            result.append(g)
        else:
            result.append(arg.vspace.zeros() if isnode(arg) else vspace(arg).zeros())
    return nw.array_from_args(*result)
nw.array_from_args.jvp = array_from_args_fwd_gradmaker
