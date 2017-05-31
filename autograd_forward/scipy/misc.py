from __future__ import absolute_import
import autograd.scipy.misc as asm

import autograd.numpy as anp

def make_fwd_grad_logsumexp(g, ans, gvs, vs, x, axis=None, b=1.0, keepdims=False):
    if not keepdims:
        if isinstance(axis, int):
            ans = anp.expand_dims(ans, axis)
        elif isinstance(axis, tuple):
            for ax in sorted(axis):
                ans = anp.expand_dims(ans, ax)
    return anp.sum(g * b * anp.exp(x - ans), axis=axis, keepdims=keepdims)
asm.logsumexp.defjvp(make_fwd_grad_logsumexp)
