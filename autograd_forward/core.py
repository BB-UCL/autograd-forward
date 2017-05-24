from __future__ import absolute_import
import autograd.core as ac
from collections import defaultdict, OrderedDict
import warnings


def make_jvp(fun, argnum=0):
    def jvp(*args, **kwargs):
        args = list(args)
        start_node = new_progenitor(args[argnum])
        args[argnum] = start_node
        def forward_mode_pass(v):
            ac.assert_vspace_match(v, start_node.vspace, None)
            start_node.forward_progenitors[start_node] = v
            active_forward_progenitors[start_node] = True
            end_node = fun(*args, **kwargs)
            active_forward_progenitors.pop(start_node)
            if not ac.isnode(end_node) or start_node not in end_node.forward_progenitors:
                warnings.warn("Output seems independent of input.")
                return end_node, ac.vspace(ac.getval(end_node)).zeros()
            return end_node, end_node.forward_progenitors[start_node]
        return forward_mode_pass, start_node
    return jvp


active_forward_progenitors = OrderedDict()

def jvp(self, argnum, ingrad, ans, gvs, vs, args, kwargs):
    try:
        return self.jvps[argnum](ingrad, ans, gvs, vs, *args, **kwargs)
    except KeyError:
        if self.jvps == {}:
            errstr = "Forward gradient of {0} not yet implemented."
        else:
            errstr = "Forward gradient of {0} w.r.t. arg number {1} not yet implemented."
        raise NotImplementedError(errstr.format(self.fun.__name__, argnum))

def defjvp(self, jvpmaker, argnum=0):
    if not hasattr(self, 'jvps'):
        self.jvps = {}
    jvpmaker.__name__ = "JVP_{}_of_{}".format(argnum, self.__name__)
    self.jvps[argnum] = jvpmaker

ac.primitive.jvp = jvp
ac.primitive.defjvp = defjvp

# class primitive(object):
#     """
#     Wraps a function so that its gradient can be specified and its invocation
#     can be recorded. For examples, see the docs."""
#     def __init__(self, fun):
#         self.fun = fun
#         self.vjps = {}
#         self.jvps = {}
#         self.zero_vjps = set()
#         self.__name__ = fun.__name__
#         self.__doc__ = fun.__doc__
#
#     def __call__(self, *args, **kwargs):
#         argvals, parents, progenitors, forward_progenitors = self.find_progenitors(args)
#         result_value = self.fun(*argvals, **kwargs)
#         if progenitors and not forward_progenitors:
#             return new_node(result_value, (self, args, kwargs, parents), progenitors, None)
#         elif progenitors and forward_progenitors:
#             result = new_node(result_value, (self, args, kwargs, parents), progenitors, dict())
#             result = self.fwd_update(args, kwargs, result, forward_progenitors)
#             return result
#         elif forward_progenitors and not progenitors:
#             result = new_node(result_value, None,                          progenitors, dict())
#             result = self.fwd_update(args, kwargs, result, forward_progenitors)
#             return result
#         else:
#             return result_value
#
#     def find_progenitors(self, args):
#         argvals = list(args)
#         parents = []
#         progenitors = set()
#         forward_progenitors = defaultdict(list)
#         for argnum, arg in enumerate(args):
#             if isnode(arg):
#                 argvals[argnum] = arg.value
#                 if argnum in self.zero_vjps: continue
#                 reverse = arg.progenitors & active_progenitors
#                 if reverse:
#                     parents.append((argnum, arg))
#                     progenitors.update(reverse)
#                 for progenitor in arg.forward_progenitors:
#                     if active_forward_progenitors.get(progenitor, False):
#                         forward_progenitors[progenitor].append((argnum, arg))
#         return argvals, parents, progenitors, forward_progenitors
#
#     def fwd_update(self, args, kwargs, result, forward_progenitors):
#         for progenitor in forward_progenitors:
#             active_forward_progenitors[progenitor] = False
#         for progenitor in active_forward_progenitors:
#             if progenitor not in forward_progenitors:
#                 continue
#             ingrads = list()
#             for argnum, arg in forward_progenitors[progenitor]:
#                 forward_grad = arg.forward_progenitors[progenitor]
#                 ingrad = self.jvp(argnum, forward_grad, result, arg.vspace,
#                                   result.vspace, args, kwargs)
#                 assert_vspace_match(ingrad, result.vspace, self, fwd=True)
#                 ingrads.append(ingrad)
#             result.forward_progenitors[progenitor] = vsum(result.vspace, *ingrads)
#             active_forward_progenitors[progenitor] = True
#         return result

def new_progenitor(x, fwd=False):
    if isnode(x):
        node = new_node(x.value, (identity, (x,), {}, [(0, x)]), x.progenitors, x.forward_progenitors)
    else:
        node = new_node(x,       (identity, (x,), {}, []      ), set(),         dict())
    if not fwd:
        node.progenitors = node.progenitors | {node}
    return node

ac.primitive_mut_add.jvp = lambda arg, g, *args : g

@ac.primitive
def identity(x) : return x
identity.defvjp(lambda g, ans, vs, gvs, x : g)

class Node(object):
    __slots__ = ['value', 'recipe', 'progenitors', 'forward_progenitors',
                 'vspace']

    def __init__(self, value, recipe, progenitors, forward_progenitors):
        self.value = value
        self.recipe = recipe
        self.progenitors = progenitors
        self.forward_progenitors = forward_progenitors
        self.vspace = vspace(value)

    def __bool__(self):
        return bool(self.value)

    __nonzero__ = __bool__

    def __str__(self):
        return "Autograd {0} with value {1} and {2} progenitors(s)".format(
            type(self).__name__, str(self.value), len(self.progenitors))


def new_node(value, recipe, progenitors, forward_progenitors):
    try:
        return node_type_mappings[type(value)](value, recipe, progenitors, forward_progenitors)
    except KeyError:
        raise TypeError("Can't differentiate w.r.t. type {}".format(type(value)))
