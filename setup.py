import glob
import os
import platform


import numpy
from distutils.core import setup, Extension
from Cython.Build import cythonize
from setuptools import find_packages
from distutils.core import Extension, setup

numpy_include_dirs = os.path.split(numpy.__file__)[0] + '/core/include'

# TODO: Appropriate g++ version pattern matching
# install g++

def set_gcc_mac():
    patterns = ['/opt/local/bin/g++-mp-[0-9]*.[0-9]*',
                '/opt/local/bin/g++-mp-[0-9]*',
                '/opt/homebrew/opt/gcc@[0-9][0-9]/bin/g++-[0-9][0-9]']
    ret = []
    for p in patterns:
        ret.extend(glob.glob(p))
    os.environ["CXX"] = ret[-1]
    os.environ["CC"] = ret[-1]

if platform.system() == 'Darwin':
    set_gcc_mac()



os.environ['CXX'] = "/opt/homebrew/opt/gcc@11/bin/g++-11"


include_dirs = [numpy_include_dirs]
compile_args = ["-fopenmp", "-std=c++14", "-O3"]
link_args = ["-fopenmp", "-std=c++14"]


ext_modules=[
    Extension("pysimpfer.simpfer",
              sources=[os.path.join("pysimpfer", "simpfer.pyx")],
              language='c++',
              include_dirs=include_dirs,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
]


setup(name="pysimpfer",
      version="0.0.1",
      packages=find_packages(),
      ext_modules=cythonize(ext_modules))
