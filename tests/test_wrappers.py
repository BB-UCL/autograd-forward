from __future__ import absolute_import
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import *
from autograd import jacobian, hessian
from autograd_forward import hessian_vector_product, jacobian_vector_product

npr.seed(1)

def test_hessian_vector_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5)
    v = npr.randn(5)
    H = hessian(fun)(a)
    check_equivalent(np.dot(H, v), hessian_vector_product(fun)(a, v))

def test_hessian_matrix_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4)
    V = npr.randn(5, 4)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V), hessian_vector_product(fun)(a, V))

def test_hessian_tensor_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4, 3)
    V = npr.randn(5, 4, 3)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V, axes=np.ndim(V)), hessian_vector_product(fun)(a, V))

def test_fwd_rev_hessian_vector_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5)
    v = npr.randn(5)
    H = hessian(fun)(a)
    check_equivalent(np.dot(H, v), hessian_vector_product(fun, method='fwd-rev')(a, v))

def test_fwd_rev_hessian_matrix_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4)
    V = npr.randn(5, 4)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V), hessian_vector_product(fun, method='fwd-rev')(a, V))

def test_fwd_rev_hessian_tensor_product():
    fun = lambda a: np.sum(np.sin(a))
    a = npr.randn(5, 4, 3)
    V = npr.randn(5, 4, 3)
    H = hessian(fun)(a)
    check_equivalent(np.tensordot(H, V, axes=np.ndim(V)), hessian_vector_product(fun, method='fwd-rev')(a, V))

def test_jacobian_vector_product():
    fun = lambda a: np.roll(np.sin(a), 1)
    a = npr.randn(5)
    V = npr.randn(5)
    J = jacobian(fun)(a)
    check_equivalent(np.dot(J, V), jacobian_vector_product(fun)(a, V))
