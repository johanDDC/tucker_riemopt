from setuptools import setup

setup(name='tucker_riemopt',
      version='0.0.1',
      description='Toolbox for riemannian optimization on manifold of tensors of fixed multilinear rank.',
      url='https://github.com/johanDDC/tucker_riemopt',
      author='Ivan Peshekhonov',
      author_email='ivan.peshekhonov@gmail.com',
      license='MIT',
      packages=['tucker_riemopt'],
      install_requires=[
          'numpy',
          'jax',
          'flax',
          'opt_einsum',
          'scipy'
      ],
      zip_safe=False)