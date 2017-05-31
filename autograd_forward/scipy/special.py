from __future__ import absolute_import

import autograd.numpy as np
import autograd.scipy.special as sp

### Gamma functions ###
sp.polygamma.defjvp(lambda g, ans, gvs, vs, n, x: g * sp.polygamma(n + 1, x), argnum=1)
sp.psi.defjvp(      lambda g, ans, gvs, vs, x: g * sp.polygamma(1, x))
sp.digamma.defjvp(  lambda g, ans, gvs, vs, x: g * sp.polygamma(1, x))
sp.gamma.defjvp(    lambda g, ans, gvs, vs, x: g * ans * sp.psi(x))
sp.gammaln.defjvp(  lambda g, ans, gvs, vs, x: g * sp.psi(x))
sp.rgamma.defjvp(   lambda g, ans, gvs, vs, x: g * sp.psi(x) / -sp.gamma(x))
sp.multigammaln.defjvp(lambda g, ans, gvs, vs, a, d:
    g * np.sum(sp.digamma(np.expand_dims(a, -1) - np.arange(d)/2.), -1))

### Bessel functions ###
sp.j0.defjvp(lambda g, ans, gvs, vs, x: -g * sp.j1(x))
sp.y0.defjvp(lambda g, ans, gvs, vs, x: -g * sp.y1(x))
sp.j1.defjvp(lambda g, ans, gvs, vs, x: g * (sp.j0(x) - sp.jn(2, x)) / 2.0)
sp.y1.defjvp(lambda g, ans, gvs, vs, x: g * (sp.y0(x) - sp.yn(2, x)) / 2.0)
sp.jn.defjvp(lambda g, ans, gvs, vs, n, x: g * (sp.jn(n - 1, x) - sp.jn(n + 1, x)) / 2.0, argnum=1)
sp.yn.defjvp(lambda g, ans, gvs, vs, n, x: g * (sp.yn(n - 1, x) - sp.yn(n + 1, x)) / 2.0, argnum=1)

### Error Function ###
sp.erf.defjvp( lambda g, ans, gvs, vs, x:  2.*g*sp.inv_root_pi*np.exp(-x**2))
sp.erfc.defjvp(lambda g, ans, gvs, vs, x: -2.*g*sp.inv_root_pi*np.exp(-x**2))

### Inverse error function ###
sp.erfinv.defjvp(lambda g, ans, gvs, vs, x: g * sp.root_pi / 2 * np.exp(sp.erfinv(x)**2))
sp.erfcinv.defjvp(lambda g, ans, gvs, vs, x: -g * sp.root_pi / 2 * np.exp(sp.erfcinv(x)**2))
