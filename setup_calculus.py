from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

# If there is an ld error, try setting LD_FLAGS="-v" to get output.
# Seems that having extra libraries in LIBRARY_PATH will cause the build to fail.
# Also, don't sudo! sudo has a different LIBRARY_PATH

#setup(
#    ext_modules=cythonize('calculus.pyx'),
#)
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("calculus", ["calculus.pyx"],
                             include_dirs=[ numpy.get_include()],
                             library_dirs=['/usr/lib/x86_64-linux-gnu'])]
)
