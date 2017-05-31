from __future__ import absolute_import

import autograd.numpy.numpy_extra as ne

ne.take.defjvp(lambda g, ans, gvs, vs, A, idx: ne.take(g, idx))
ne.untake.defjvp(lambda g, ans, gvs, vs, x, idx, template : ne.untake(g, idx, template))
