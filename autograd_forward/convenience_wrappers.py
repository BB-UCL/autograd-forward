"""Convenience functions built on top of `grad`."""
from __future__ import absolute_import

from autograd.convenience_wrappers import (attach_name_and_doc, safe_type,
                                           cast_to_same_dtype)

from autograd_forward.core import make_jvp


def forward_derivative(fun, argnum=0):
    """
    Derivative of fun w.r.t. scalar argument argnum.
    """
    @attach_name_and_doc(fun, argnum, 'Forward mode derivative')
    def dervfun(*args, **kwargs):
        args = list(args)
        args[argnum] = safe_type(args[argnum])
        jvp, start_node = make_jvp(fun, argnum)(*args, **kwargs)
        ans, d = jvp(cast_to_same_dtype(1.0, args[argnum]))
        return d
    return dervfun

# def jacobian(fun, argnum=0):
#     """
#     Returns a function which computes the Jacobian of `fun` with respect to
#     positional argument number `argnum`, which must be a scalar or array. Unlike
#     `grad` it is not restricted to scalar-output functions, but also it cannot
#     take derivatives with respect to some argument types (like lists or dicts).
#     If the input to `fun` has shape (in1, in2, ...) and the output has shape
#     (out1, out2, ...) then the Jacobian has shape (out1, out2, ..., in1, in2, ...).
#     """
#     def getshape(val):
#         val = getval(val)
#         assert np.isscalar(val) or isinstance(val, np.ndarray), \
#             'Jacobian requires input and output to be scalar- or array-valued'
#         return np.shape(val)
#
#     def unit_vectors(shape):
#         for idxs in it.product(*map(range, shape)):
#             vect = np.zeros(shape)
#             vect[idxs] = 1
#             yield vect
#
#     concatenate = lambda lst: np.concatenate(map(np.atleast_1d, lst))
#
#     @attach_name_and_doc(fun, argnum, 'Jacobian')
#     def jacfun(*args, **kwargs):
#         vjp, ans = make_vjp(fun, argnum)(*args, **kwargs)
#         outshape = getshape(ans)
#         grads = map(vjp, unit_vectors(outshape))
#         jacobian_shape = outshape + getshape(args[argnum])
#         return np.reshape(concatenate(grads), jacobian_shape)
#
#     return jacfun

# def hessian_vector_product(fun, argnum=0, method='rev-rev'):
#     """Builds a function that returns the exact Hessian-vector product.
#     The returned function has arguments (*args, vector, **kwargs), and takes
#     roughly 4x as long to evaluate as the original function.
#
#     There are two methods available, specified by the `method' parameter:
#     rev-rev (default) and fwd-rev. fwd-rev is faster and has lower memory
#     overhead but is incompatible with some primitives."""
#     if method == 'rev-rev':
#         fun_grad = grad(fun, argnum)
#         def vector_dot_grad(*args, **kwargs):
#             args, vector = args[:-1], args[-1]
#             return np.tensordot(fun_grad(*args, **kwargs), vector, np.ndim(vector))
#         return grad(vector_dot_grad, argnum)  # Grad wrt original input.
#     elif method == 'fwd-rev':
#         return jacobian_vector_product(grad(fun, argnum), argnum)
#     else:
#         raise ValueError("{} is not a valid method for hessian_vector_product. "
#                          "Valid methods are: 'rev-rev', 'fwd-rev'.".format(method))

def jacobian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Jacobian-vector product, that
    is the Jacobian matrix right-multiplied by vector. The returned function
    has arguments (*args, vector, **kwargs)."""
    jvp = make_jvp(fun, argnum=argnum)
    def jac_vec_prod(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return jvp(*args, **kwargs)[0](vector)[1]
    return jac_vec_prod
