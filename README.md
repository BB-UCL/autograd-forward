# Autograd: forward! [![Build Status](https://travis-ci.org/BB-UCL/autograd-forward.svg?branch=master)](https://travis-ci.org/BB-UCL/autograd-forward)
This package adds forward mode differentiation to the already fantastic [Autograd](https://github.com/HIPS/autograd). Efficient Jacobian vector product computation (analagous to Theano's Rop) and a more efficient Hessian vector product are included, and fully compatible with Autograd's grad operator, as well as its other [convenience wrappers](https://github.com/HIPS/autograd/blob/master/autograd/convenience_wrappers.py).

This package was written and is maintained by Jamie Townsend.
