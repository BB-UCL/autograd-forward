from __future__ import absolute_import
import autograd.scipy.stats.dirichlet as di
import autograd.numpy as np
from autograd.scipy.special import digamma

di.logpdf.defjvp(lambda g, ans, gvs, vs, x, alpha: np.inner(g, (alpha - 1) / x), argnum=0)
di.logpdf.defjvp(lambda g, ans, gvs, vs, x, alpha: np.inner(g, (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x))), argnum=1)

di.pdf.defjvp(lambda g, ans, gvs, vs, x, alpha: np.inner(g, ans * (alpha - 1) / x), argnum=0)
di.pdf.defjvp(lambda g, ans, gvs, vs, x, alpha: np.inner(g, ans * (digamma(np.sum(alpha)) - digamma(alpha) + np.log(x))), argnum=1)
