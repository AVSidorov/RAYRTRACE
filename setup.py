from setuptools import setup, Extension

# from distutils.core import setup, Extension

module1 = Extension('sidtrace',
                    sources=['sidtracemodule.c'],
                    include_dirs=['C:/Program Files/Python38/Lib/site-packages/numpy/core/include/numpy'])

setup(name='trace',
      version='0.1',
      description='This is a trace cycle function',
      ext_modules=[module1])
