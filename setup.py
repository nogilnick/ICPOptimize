import os
from   setuptools          import setup, Extension
from   subprocess          import check_output
from   distutils.sysconfig import get_python_inc
import numpy               as     np

incdir = os.path.join(get_python_inc(plat_specific=1))

# Check if gcc is installed
compileFlags = (['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '-O3', '-march=native'],
                ['/DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION', '/Ox'])

try:
   check_output(['gcc', '-v'])
   compileFlags = compileFlags[0]
except:  # Try cl flags
   compileFlags = compileFlags[1]
   
# Read in readme
with open('README.md') as F:
   desc = F.read()
   
module = Extension(
   'ICP.PathSearch',
   include_dirs=[incdir, np.get_include()],
   libraries=[],
   library_dirs=[],
   sources=[os.path.join('ICP', 'PathSearch.c')],
   extra_compile_args=compileFlags)

setup(
   name='ICPOptimize',
   version='1.3',
   description='Python 3 Implementation of ICP and ICPRE',
   author='Nicholas T. Smith',
   author_email='nicholastsmithblog@gmail.com',
   url="https://github.com/nicholastoddsmith/ICPOptimize",
   long_description_content_type="text/markdown",
   long_description=desc,
   packages=['ICP'],
   ext_modules=[module],
   keywords=["optimization", "optimizer", "linear", "ICP", "ICPRE"],
   classifiers=[
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Mathematics',
      'Natural Language :: English',
      'Operating System :: OS Independent',
      'Programming Language :: C',
      'Programming Language :: Cython',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3']
   )
