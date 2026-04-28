"""
Setup script for compiling Cython extension.

Usage:
    pip install cython
    python3 setup_cython.py build_ext --inplace
    
This creates decide_cython.cpython-*.so which can be imported:
    from decide_cython import compute_coop_prob_cython, decide_cython
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="decide_cython",
    ext_modules=cythonize(
        "decide_cython.pyx",
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)
