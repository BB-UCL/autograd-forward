# Autograd: forward! [![Build Status](https://travis-ci.org/BB-UCL/autograd-forward.svg?branch=master)](https://travis-ci.org/BB-UCL/autograd-forward)
This package adds forward mode differentiation to the already fantastic [Autograd](https://github.com/HIPS/autograd). Efficient Jacobian vector product computation (analagous to Theano's Rop) and a more efficient Hessian vector product are included, and fully compatible with Autograd's grad operator, as well as its other [convenience wrappers](https://github.com/HIPS/autograd/blob/master/autograd/convenience_wrappers.py).

Autograd-forward enables you to do things like:
```python
In [1]: from autograd_forward import jacobian_vector_product

In [2]: import autograd.numpy as np

In [3]: def f(x):
   ...:     return x**2 + 1
   ...:

In [4]: jvp = jacobian_vector_product(f)

In [5]: x = np.array([1., 2., 3.])

In [6]: v = np.array([4., 5., 6.])

In [7]: jvp(x, v)
Out[7]: array([  8.,  20.,  36.])
```
Mixing forward mode with Autograd's reverse mode operators 'just works':
```python
In [8]: from autograd import grad

In [9]: scalar_output_fun = lambda x, v: np.sum(jvp(x, v))

In [10]: grad(scalar_output_fun)(x, v)
Out[10]: array([  8.,  10.,  12.])
```
For functions which output a scalar, you can calculate _Hessian vector products_ by doing:
```python
In [11]: def g(x):
   ...:     return np.sum(x**3)
   ...:

In [12]: hvp = jacobian_vector_product(grad(g))

In [13]: hvp(x, v)
Out[13]: array([  24.,   60.,  108.])
```
Or you can use `autograd_forward.hessian_vector_product`, with the `mode` keyword argument set to `fwd-rev`.

This package was written and is maintained by Jamie Townsend.

# Installation
Right now, autograd-forward depends on the latest bleeding edge version of Autograd on Github. You can install this version of Autograd from Github by doing
```
pip install --upgrade git+https://github.com/HIPS/autograd.git
```
You can then install autograd-forward with
```
pip install --upgrade git+https://github.com/BB-UCL/autograd-forward.git
```

# Supported primitive operations
I've so far implemented forward derivatives for all of the Numpy primitives covered by Autograd, except those in `numpy.linalg`, and I've also implemented some of the Scipy primitives. Please file an issue if there's something that you need to differentiate that isn't yet implemented.
