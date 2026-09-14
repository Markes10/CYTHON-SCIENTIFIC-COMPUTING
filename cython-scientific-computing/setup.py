from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "sparse_solver",
        ["src/sparse_solver.pyx"],
        include_dirs=[np.get_include(), "src"],
        extra_compile_args=["-O3", "-ffast-math"],
    )
]

setup(
    name="cython-scientific-computing",
    version="1.0.0",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
