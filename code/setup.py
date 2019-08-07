from setuptools import setup, find_packages

__version__ = "0.1"


setup(name='timage',
      version=__version__,
      description='',
      url='',
      packages=find_packages(),
      install_requires=['Click'],
      entry_points='''
          [console_scripts]
          timage=timage.__main__:cli
      ''',
      )
