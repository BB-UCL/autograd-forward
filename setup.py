from setuptools import setup

setup(
    name='autograd-forward',
    version='0.1.0',
    description='Autograd extension for forward mode autodiff',
    author='James Townsend',
    author_email="james.townsend@cs.ucl.ac.uk",
    packages=['autograd_forward', 'autograd_forward.numpy',
              'autograd_forward.scipy', 'autograd_forward.scipy.stats'],
    install_requires=['autograd', 'numpy>=1.12', 'scipy>=0.17', 'future>=0.15.2'],
    keywords=['Automatic differentiation', 'backpropagation', 'gradients',
              'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy', 'Scipy'],
    url='https://github.com/BB-UCL/autograd-forward',
    license='MIT',
    classifiers=['Development Status :: 4 - Beta',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6'],
)
