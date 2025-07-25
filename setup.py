from setuptools import setup, Extension, find_namespace_packages
from Cython.Build import cythonize

extensions = [
    Extension(
        name="sdiff.cython._shared",
        sources=["sdiff/cython/_shared.c"],
    ),
    Extension(
        name="sdiff.cython.compare",
        sources=["sdiff/cython/compare.pyx"],
        include_dirs=["sdiff/cython"],  # Allow cimport to find .pxd files
    ),
    Extension(
        name="sdiff.cython.cmyers",
        sources=["sdiff/cython/cmyers.pyx"],
        include_dirs=["sdiff/cython"],  # Allow cimport to find .pxd files
    )
]

setup(
    name="sdiff",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
        },
        include_path=["sdiff/cython"],
        annotate=True,
        shared_utility_qualified_name="sdiff.cython._shared",
    ),
    packages=find_namespace_packages(),
    zip_safe=False,
    include_package_data=True,
)
