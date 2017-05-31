from __future__ import absolute_import

# Try block needed in case the user has an
# old version of scipy without multivariate normal.
try:
    from . import dirichlet
except AttributeError:
    pass
