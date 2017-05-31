from __future__ import absolute_import
import autograd.numpy.numpy_grads as npg
import autograd.numpy as anp
from autograd.core import getval
from builtins import range


# ----- Functions that are constant w.r.t. continuous inputs -----
anp.nan_to_num.defjvp(lambda g, ans, gvs, vs, x: anp.where(anp.isfinite(x), g, 0.))

# ----- Binary ufuncs -----
anp.add.defjvp(        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
anp.add.defjvp(        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g), argnum=1)
anp.multiply.defjvp(   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, y * g))
anp.multiply.defjvp(   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, x * g), argnum=1)
anp.subtract.defjvp(   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
anp.subtract.defjvp(   lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, -g), argnum=1)
anp.divide.defjvp(     lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs,   g / y))
anp.divide.defjvp(     lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, - g * x / y**2), argnum=1)
anp.maximum.defjvp(    lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(x, ans, y)))
anp.maximum.defjvp(    lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(y, ans, x)), argnum=1)
anp.minimum.defjvp(    lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(x, ans, y)))
anp.minimum.defjvp(    lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(y, ans, x)), argnum=1)
anp.fmax.defjvp(       lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(x, ans, y)))
anp.fmax.defjvp(       lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(y, ans, x)), argnum=1)
anp.fmin.defjvp(       lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(x, ans, y)))
anp.fmin.defjvp(       lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * npg.balanced_eq(y, ans, x)), argnum=1)
anp.logaddexp.defjvp(  lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * anp.exp(x-ans)))
anp.logaddexp.defjvp(  lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * anp.exp(y-ans)), argnum=1)
anp.logaddexp2.defjvp( lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * 2**(x-ans)))
anp.logaddexp2.defjvp( lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g * 2**(y-ans)), argnum=1)
anp.true_divide.defjvp(lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g / y))
anp.true_divide.defjvp(lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, - g * x / y**2), argnum=1)
anp.mod.defjvp(        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
anp.remainder.defjvp(  lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, g))
anp.mod.defjvp(        lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, -g * anp.floor(x/y)), argnum=1)
anp.remainder.defjvp(  lambda g, ans, gvs, vs, x, y : broadcast(gvs, vs, -g * anp.floor(x/y)), argnum=1)
anp.power.defjvp(
    lambda g, ans, gvs, vs, x, y :
    broadcast(gvs, vs, g * y * x ** anp.where(y, y - 1, 1.)))
anp.power.defjvp(
    lambda g, ans, gvs, vs, x, y :
    broadcast(gvs, vs, g * anp.log(npg.replace_zero(x, 1.)) * x ** y), argnum=1)

# ----- Simple grads -----
anp.negative.defjvp(lambda g, ans, gvs, vs, x: -g)
anp.abs.defjvp(
    lambda g, ans, gvs, vs, x : anp.real(g * npg.replace_zero(anp.conj(x), 0.)) / npg.replace_zero(ans, 1.))
anp.fabs.defjvp(    lambda g, ans, gvs, vs, x : anp.sign(x) * g)  # fabs doesn't take complex numbers.
anp.absolute.defjvp(lambda g, ans, gvs, vs, x : anp.real(g * anp.conj(x)) / ans)
anp.reciprocal.defjvp(lambda g, ans, gvs, vs, x : - g / x**2)
anp.exp.defjvp(   lambda g, ans, gvs, vs, x : ans * g)
anp.exp2.defjvp(  lambda g, ans, gvs, vs, x : ans * anp.log(2) * g)
anp.expm1.defjvp( lambda g, ans, gvs, vs, x : (ans + 1) * g)
anp.log.defjvp(   lambda g, ans, gvs, vs, x : g / x)
anp.log2.defjvp(  lambda g, ans, gvs, vs, x : g / x / anp.log(2))
anp.log10.defjvp( lambda g, ans, gvs, vs, x : g / x / anp.log(10))
anp.log1p.defjvp( lambda g, ans, gvs, vs, x : g / (x + 1))
anp.sin.defjvp(   lambda g, ans, gvs, vs, x : g * anp.cos(x))
anp.cos.defjvp(   lambda g, ans, gvs, vs, x : - g * anp.sin(x))
anp.tan.defjvp(   lambda g, ans, gvs, vs, x : g / anp.cos(x) **2)
anp.arcsin.defjvp(lambda g, ans, gvs, vs, x : g / anp.sqrt(1 - x**2))
anp.arccos.defjvp(lambda g, ans, gvs, vs, x :-g / anp.sqrt(1 - x**2))
anp.arctan.defjvp(lambda g, ans, gvs, vs, x : g / (1 + x**2))
anp.sinh.defjvp(  lambda g, ans, gvs, vs, x : g * anp.cosh(x))
anp.cosh.defjvp(  lambda g, ans, gvs, vs, x : g * anp.sinh(x))
anp.tanh.defjvp(  lambda g, ans, gvs, vs, x : g / anp.cosh(x) **2)
anp.arcsinh.defjvp(lambda g, ans, gvs, vs, x : g / anp.sqrt(x**2 + 1))
anp.arccosh.defjvp(lambda g, ans, gvs, vs, x : g / anp.sqrt(x**2 - 1))
anp.arctanh.defjvp(lambda g, ans, gvs, vs, x : g / (1 - x**2))
anp.rad2deg.defjvp(lambda g, ans, gvs, vs, x : g / anp.pi * 180.0)
anp.degrees.defjvp(lambda g, ans, gvs, vs, x : g / anp.pi * 180.0)
anp.deg2rad.defjvp(lambda g, ans, gvs, vs, x : g * anp.pi / 180.0)
anp.radians.defjvp(lambda g, ans, gvs, vs, x : g * anp.pi / 180.0)
anp.square.defjvp( lambda g, ans, gvs, vs, x : g * 2 * x)
anp.sqrt.defjvp(   lambda g, ans, gvs, vs, x : g * 0.5 * x**-0.5)
anp.sinc.defjvp(   lambda g, ans, gvs, vs, x : g * (anp.cos(anp.pi*x)*anp.pi*x - anp.sin(anp.pi*x))/(anp.pi*x**2))
anp.reshape.defjvp(lambda g, ans, gvs, vs, x, shape, order=None : anp.reshape(g, vs.shape, order=order))
anp.roll.defjvp(   lambda g, ans, gvs, vs, x, shift, axis=None  : anp.roll(g, shift, axis=axis))
anp.array_split.defjvp(lambda g, ans, gvs, vs, ary, idxs, axis=0 : anp.array_split(g, idxs, axis=axis))
anp.split.defjvp(      lambda g, ans, gvs, vs, ary, idxs, axis=0 : anp.split(g, idxs, axis=axis))
anp.vsplit.defjvp(     lambda g, ans, gvs, vs, ary, idxs         : anp.vsplit(g, idxs))
anp.hsplit.defjvp(     lambda g, ans, gvs, vs, ary, idxs         : anp.hsplit(g, idxs))
anp.dsplit.defjvp(     lambda g, ans, gvs, vs, ary, idxs         : anp.dsplit(g, idxs))
anp.ravel.defjvp(  lambda g, ans, gvs, vs, x, order=None   : anp.ravel(g, order=order))
anp.expand_dims.defjvp(lambda g, ans, gvs, vs, x, axis     : anp.expand_dims(g, axis))
anp.squeeze.defjvp(lambda g, ans, gvs, vs, x, axis=None    : anp.squeeze(g, axis))
anp.diag.defjvp(   lambda g, ans, gvs, vs, x, k=0          : anp.diag(g, k))
anp.flipud.defjvp( lambda g, ans, gvs, vs, x,              : anp.flipud(g))
anp.fliplr.defjvp( lambda g, ans, gvs, vs, x,              : anp.fliplr(g))
anp.rot90.defjvp(  lambda g, ans, gvs, vs, x, k=1          : anp.rot90(g, k))
anp.trace.defjvp(  lambda g, ans, gvs, vs, x, offset=0     : anp.trace(g, offset))
anp.full.defjvp(   lambda g, ans, gvs, vs, shape, fill_value, dtype=None : anp.full(shape, g, dtype), argnum=1)
anp.triu.defjvp(   lambda g, ans, gvs, vs, x, k=0          : anp.triu(g, k=k))
anp.tril.defjvp(   lambda g, ans, gvs, vs, x, k=0          : anp.tril(g, k=k))
anp.clip.defjvp(   lambda g, ans, gvs, vs, x, a_min, a_max : g * anp.logical_and(ans != a_min, ans != a_max))
anp.swapaxes.defjvp(lambda g, ans, gvs, vs, x, axis1, axis2: anp.swapaxes(g, axis1, axis2))
anp.rollaxis.defjvp(lambda g, ans, gvs, vs, a, axis, start=0: anp.rollaxis(g, axis, start))
anp.real_if_close.defjvp(lambda g, ans, gvs, vs, x : npg.match_complex(vs, g))
anp.real.defjvp(  lambda g, ans, gvs, vs, x   : anp.real(g))
anp.imag.defjvp(  lambda g, ans, gvs, vs, x   : npg.match_complex(vs, -1j * g))
anp.conj.defjvp(  lambda g, ans, gvs, vs, x   : anp.conj(g))
anp.angle.defjvp( lambda g, ans, gvs, vs, x   : npg.match_complex(vs, g * anp.conj(x * 1j) / anp.abs(x)**2))
anp.where.defjvp( lambda g, ans, gvs, vs, c, x=None, y=None : anp.where(c, g, anp.zeros(anp.shape(g))), argnum=1)
anp.where.defjvp( lambda g, ans, gvs, vs, c, x=None, y=None : anp.where(c, anp.zeros(g.shape), g), argnum=2)
anp.cross.defjvp(lambda g, ans, gvs, vs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(g, b, axisa, axisb, axisc, axis), argnum=0)
anp.cross.defjvp(lambda g, ans, gvs, vs, a, b, axisa=-1, axisb=-1, axisc=-1, axis=None :
                  anp.cross(a, g, axisa, axisb, axisc, axis), argnum=1)

# ----- Trickier grads -----
anp.diff.defjvp(lambda g, ans, gvs, vs, a, n=1, axis=-1: anp.diff(g, n, axis))
anp.repeat.defjvp(lambda g, ans, gvs, vs, x, repeats, axis=None: anp.repeat(g, repeats, axis))
anp.tile.defjvp(lambda g, ans, gvs, vs, x, reps: anp.tile(g, reps))
anp.kron.defjvp(lambda g, ans, gvs, vs, a, b: anp.kron(g, b))
anp.kron.defjvp(lambda g, ans, gvs, vs, a, b: anp.kron(a, g), argnum=1)
anp.transpose.defjvp(lambda g, ans, gvs, vs, x, axes=None: anp.transpose(g, axes))
anp.sum.defjvp(lambda g, ans, gvs, vs, x, axis=None, keepdims=False: anp.sum(g, axis=axis, keepdims=keepdims))
anp.mean.defjvp(lambda g, ans, gvs, vs, x, axis=None, keepdims=False: anp.mean(g, axis=axis, keepdims=keepdims))
anp.prod.defjvp(lambda g, ans, gvs, vs, x, axis=None, keepdims=False: ans * anp.sum(g / x, axis=axis, keepdims=keepdims))

def forward_grad_np_var(g, ans, gvs, vs, x, axis=None, ddof=0, keepdims=False):
    if vs.iscomplex:
        g = g + 0j
    if axis is None:
        if gvs.iscomplex:
            num_reps = gvs.size / 2
        else:
            num_reps = gvs.size
    elif isinstance(axis, int):
        num_reps = gvs.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(gvs.shape)[list(axis)])

    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (2.0 * anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            (num_reps - ddof))
anp.var.defjvp(forward_grad_np_var)

def forward_grad_np_std(g, ans, gvs, vs, x, axis=None, ddof=0, keepdims=False):
    if axis is None:
        if gvs.iscomplex:
            num_reps = gvs.size / 2
        else:
            num_reps = gvs.size
    elif isinstance(axis, int):
        num_reps = gvs.shape[axis]
    elif isinstance(axis, tuple):
        num_reps = anp.prod(anp.array(gvs.shape)[list(axis)])

    if num_reps <= 1:
        return vs.zeros()
    x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
    return (anp.sum(anp.real(g * x_minus_mean), axis=axis, keepdims=keepdims) /
            ((num_reps - ddof) * ans))
anp.std.defjvp(forward_grad_np_std)

def fwd_grad_chooser(g, ans, gvs, vs, x, axis=None, keepdims=False):
    if anp.isscalar(x):
        return g
    if not keepdims:
        if isinstance(axis, int):
            ans = anp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = anp.expand_dims(ans, ax)
    chosen_locations = x == ans
    return anp.sum(g * chosen_locations, axis=axis, keepdims=keepdims)

anp.max.defjvp(fwd_grad_chooser)
anp.min.defjvp(fwd_grad_chooser)
anp.amax.defjvp(fwd_grad_chooser)
anp.amin.defjvp(fwd_grad_chooser)

anp.cumsum.defjvp(lambda g, ans, gvs, vs, x, axis=None: anp.cumsum(g, axis=axis))

anp.inner.defjvp(lambda g, ans, gvs, vs, A, B: anp.inner(g, B))
anp.inner.defjvp(lambda g, ans, gvs, vs, A, B: anp.inner(A, g), argnum=1)

anp.matmul.defjvp(lambda g, ans, gvs, vs, A, B: anp.matmul(g, B))
anp.matmul.defjvp(lambda g, ans, gvs, vs, A, B: anp.matmul(A, g), argnum=1)

anp.dot.defjvp(lambda g, ans, gvs, vs, A, B: anp.dot(g, B))
anp.dot.defjvp(lambda g, ans, gvs, vs, A, B: anp.dot(A, g), argnum=1)

anp.tensordot.defjvp(lambda g, ans, gvs, vs, A, B, axes=2: anp.tensordot(g, B, axes=axes))
anp.tensordot.defjvp(lambda g, ans, gvs, vs, A, B, axes=2: anp.tensordot(A, g, axes=axes), argnum=1)

anp.outer.defjvp(lambda g, ans, gvs, vs, a, b : anp.outer(g, b))
anp.outer.defjvp(lambda g, ans, gvs, vs, a, b : anp.outer(a, g), argnum=1)

def fwd_grad_concatenate_args(argnum, g, ans, gvs, vs, axis_args, kwargs):
    result = []
    for i in range(1, len(axis_args)):
        if i == argnum:
            result.append(g)
        else:
            result.append(anp.zeros_like(getval(axis_args[i])))
    return anp.concatenate_args(axis_args[0], *result)
anp.concatenate_args.jvp = fwd_grad_concatenate_args

def fwd_grad_sort(g, ans, gvs, vs, x, axis=-1, kind='quicksort', order=None):
    sort_perm = anp.argsort(x, axis, kind, order)
    return g[sort_perm]
anp.sort.defjvp(fwd_grad_sort)
anp.msort.defjvp(lambda g, ans, gvs, vs, x: fwd_grad_sort(g, ans, gvs, vs, x,
                                                          axis=0))

def fwd_grad_partition(g, ans, gvs, vs, x, kth, axis=-1, kind='introselect', order=None):
    partition_perm = anp.argpartition(x, kth, axis, kind, order)
    return g[partition_perm]
anp.partition.defjvp(fwd_grad_partition)

def atleast_jvpmaker(fun):
    def jvp(g, ans, gvs, vs, *arys):
        if len(arys) > 1:
            raise NotImplementedError("Can't handle multiple arguments yet.")
        return fun(g)
    return jvp
anp.atleast_1d.defjvp(atleast_jvpmaker(anp.atleast_1d))
anp.atleast_2d.defjvp(atleast_jvpmaker(anp.atleast_2d))
anp.atleast_3d.defjvp(atleast_jvpmaker(anp.atleast_3d))

def fwd_grad_einsum(argnum, g, ans, gvs, vs, operands, kwargs):
    operands = list(operands)
    operands[argnum] = g
    return anp.einsum(*operands)
anp.einsum.jvp = fwd_grad_einsum

def broadcast(gvs, vs, result, broadcast_idx=0):
    while anp.ndim(result) < len(vs.shape):
        result = anp.expand_dims(result, 0)
    for axis, size in enumerate(anp.shape(result)):
        if size == 1:
            result = anp.repeat(result, vs.shape[axis], axis=axis)
    if vs.iscomplex and not gvs.iscomplex:
        result = result + 0j
    return result
