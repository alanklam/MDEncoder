from setuptools import setup, find_packages

setup(name='MDEncoder',
      version='0.1',
      description='Analysis tool for molecular simulations using deep neural network.',
      url='',
      author='Kin Lam',
      author_email='',
      license='MIT',
      packages=find_packages(),
      install_requires=[
              "keras",
              "tensorflow",
              "numpy",
              "sklearn",
              "matplotlib"
      ],
      zip_safe=False)
