from __future__ import absolute_import

from autograd.convenience_wrappers import (attach_name_and_doc, safe_type,
                                           cast_to_same_dtype, grad)
from autograd.convenience_wrappers import hessian_vector_product as ahvp
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

def hessian_vector_product(fun, argnum=0, method='rev-rev'):
    """Builds a function that returns the exact Hessian-vector product.
    The returned function has arguments (*args, vector, **kwargs), and takes
    roughly 4x as long to evaluate as the original function.

    There are two methods available, specified by the `method' parameter:
    rev-rev (default) and fwd-rev. fwd-rev is faster and has lower memory
    overhead but is incompatible with some primitives."""
    if method == 'rev-rev':
        return ahvp(fun, argnum)
    elif method == 'fwd-rev':
        return jacobian_vector_product(grad(fun, argnum), argnum)
    else:
        raise ValueError("{} is not a valid method for hessian_vector_product. "
                         "Valid methods are: 'rev-rev', 'fwd-rev'.".format(method))

def jacobian_vector_product(fun, argnum=0):
    """Builds a function that returns the exact Jacobian-vector product, that
    is the Jacobian matrix right-multiplied by vector. The returned function
    has arguments (*args, vector, **kwargs)."""
    jvp = make_jvp(fun, argnum=argnum)
    def jac_vec_prod(*args, **kwargs):
        args, vector = args[:-1], args[-1]
        return jvp(*args, **kwargs)[0](vector)[1]
    return jac_vec_prod
