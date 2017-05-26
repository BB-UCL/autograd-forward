from __future__ import absolute_import
import autograd.core as ac
from collections import defaultdict, OrderedDict
import warnings

def make_jvp(fun, argnum=0):
    def jvp(*args, **kwargs):
        args = list(args)
        start_node = new_progenitor(args[argnum], fwd=True)
        args[argnum] = start_node
        def forward_mode_pass(v):
            ac.assert_vspace_match(v, start_node.vspace, None)
            start_node.progenitors[1][start_node] = v
            active_forward_progenitors[start_node] = True
            end_node = fun(*args, **kwargs)
            active_forward_progenitors.pop(start_node)
            if not ac.isnode(end_node) or start_node not in end_node.progenitors[1]:
                warnings.warn("Output seems independent of input.")
                return end_node, ac.vspace(ac.getval(end_node)).zeros()
            return end_node, end_node.progenitors[1][start_node]
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

def primitive_call(self, *args, **kwargs):
    argvals, parents, progenitors, forward_progenitors = self.find_progenitors(args)
    result_value = self.fun(*argvals, **kwargs)
    if progenitors and not forward_progenitors:
        return ac.new_node(result_value, (self, args, kwargs, parents), (progenitors, dict()))
    elif progenitors and forward_progenitors:
        result = ac.new_node(result_value, (self, args, kwargs, parents), (progenitors, dict()))
        result = self.fwd_update(args, kwargs, result, forward_progenitors)
        return result
    elif forward_progenitors and not progenitors:
        result = ac.new_node(result_value, None,                          (progenitors, dict()))
        result = self.fwd_update(args, kwargs, result, forward_progenitors)
        return result
    else:
        return result_value

def find_progenitors(self, args):
    argvals = list(args)
    parents = []
    rev_progenitors = set()
    fwd_progenitors = defaultdict(list)
    for argnum, arg in enumerate(args):
        if ac.isnode(arg):
            if argnum in self.zero_vjps: continue
            argvals[argnum] = arg.value

            arg_rev_progenitors, arg_fwd_progenitors = arg.progenitors
            for progenitor in arg_fwd_progenitors:
                if active_forward_progenitors.get(progenitor, False):
                    fwd_progenitors[progenitor].append((argnum, arg))

            reverse = arg_rev_progenitors & ac.active_progenitors
            if reverse:
                parents.append((argnum, arg))
                rev_progenitors.update(reverse)

    return argvals, parents, rev_progenitors, fwd_progenitors

def fwd_update(self, args, kwargs, result, forward_progenitors):
    for progenitor in forward_progenitors:
        active_forward_progenitors[progenitor] = False
    for progenitor in active_forward_progenitors:
        if progenitor not in forward_progenitors:
            continue
        total_ingrad = None
        for argnum, arg in forward_progenitors[progenitor]:
            forward_grad = arg.progenitors[1][progenitor]
            ingrad = self.jvp(argnum, forward_grad, result, arg.vspace,
                              result.vspace, args, kwargs)
            assert_vspace_match(ingrad, result.vspace, self, fwd=True)
            total_ingrad = ac.add_outgrads(result.vspace, total_ingrad, ingrad)
        result.progenitors[1][progenitor] = total_ingrad[0]
        active_forward_progenitors[progenitor] = True
    return result

ac.primitive.__call__ = primitive_call
ac.primitive.find_progenitors = find_progenitors
ac.primitive.fwd_update = fwd_update

def new_progenitor(x, fwd=False):
    if ac.isnode(x):
        node = ac.new_node(x.value, (ac.identity, (x,), {}, [(0, x)]), x.progenitors)
    else:
        node = ac.new_node(x,       (ac.identity, (x,), {}, []      ), (set(),         dict()               ))
    if not fwd:
        node.progenitors[0].add(node)
    return node
ac.new_progenitor = new_progenitor

ac.primitive_mut_add.jvp = lambda arg, g, *args : g

def assert_vspace_match(x, expected_vspace, fun, fwd=False):
    grad_string = "Forward grad" if fwd else "Grad"
    assert expected_vspace == ac.vspace(ac.getval(x)), \
        "\n{} of {} returned unexpected vector space" \
        "\nVector space is {}" \
        "\nExpected        {}".format(grad_string, fun, ac.vspace(ac.getval(x)), expected_vspace)
