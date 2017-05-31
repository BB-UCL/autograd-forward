from __future__ import absolute_import
import autograd.container_types as ct
from autograd.core import vspace, isnode


ct.sequence_take.defjvp(lambda g, ans, gvs, vs, A, idx: ct.sequence_take(g, idx))
def fwd_grad_sequence_extend_right(argnum, g, ans, gvs, vs, args, kwargs):
    zeros = list(arg.vspace.zeros() if isnode(arg) else vspace(arg).zeros()
                 for arg in args)
    zeros[argnum] = g
    return ct.sequence_extend_right(*zeros)
ct.sequence_extend_right.jvp = fwd_grad_sequence_extend_right

def fwd_grad_sequence_extend_left(argnum, g, ans, gvs, vs, args, kwargs):
    zeros = list(arg.vspace.zeros() if isnode(arg) else vspace(arg).zeros()
                 for arg in args)
    zeros[argnum] = g
    return ct.sequence_extend_left(*zeros)
ct.sequence_extend_left.jvp = fwd_grad_sequence_extend_left

ct.sequence_untake.defjvp(lambda g, ans, gvs, vs, x, idx, template : ct.sequence_untake(g, idx, vs))

def fwd_grad_make_sequence(argnum, g, ans, gvs, vs, args, kwargs):
    typ, elts = args[0], args[1:]
    zeros = list(elt.vspace.zeros() if isnode(elt) else vspace(elt).zeros()
                 for elt in elts)
    zeros[argnum - 1] = g
    return ct.make_sequence(typ, *zeros)
ct.make_sequence.jvp = fwd_grad_make_sequence

ct.dict_take.defjvp(lambda g, ans, gvs, vs, A, idx: g[idx])
