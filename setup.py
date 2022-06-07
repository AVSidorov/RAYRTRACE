from setuptools import setup, Extension
import numpy as np

# from distutils.core import setup, Extension

module1 = Extension('sidtrace',
                    sources=['sidtracemodule.c'],
                    include_dirs=[np.get_include() + r"\numpy"])

setup(name='trace',
      version='0.1',
      description='This is a trace cycle function',
      ext_modules=[module1])
