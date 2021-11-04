from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deep_quintic'))

VERSION = 0.1

setup(name='deep_quintic',
      version=VERSION,
      description='DeepQuintic',
      author='Marc Bestmann',
      author_email='marc.bestmann@uni-hamburg.de',
      license='',
      zip_safe=False,
      install_requires=[])

