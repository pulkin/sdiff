import os
import sys

from distutils.core import Distribution, Extension

import Cython
from Cython.Build.Dependencies import cythonize
from Cython.Build.Cache import get_cython_cache_dir

from Cython.Build.Inline import cython_inline, _inline_key, strip_common_indent, _get_build_extension, load_dynamic


def build_inline_module(code,
                  lib_dir=os.path.join(get_cython_cache_dir(), 'inline'),
                  cython_include_dirs=None, cython_compiler_directives=None,
                  force=False, quiet=False, annotate=False, language_level=None):

    cython_compiler_directives = dict(cython_compiler_directives) if cython_compiler_directives else {}
    if language_level is None and 'language_level' not in cython_compiler_directives:
        language_level = '3'
    if language_level is not None:
        cython_compiler_directives['language_level'] = language_level

    code = strip_common_indent(code)

    key_hash = _inline_key(code, None, language_level)
    module_name = "_cython_inline_" + key_hash

    if module_name in sys.modules:
        module = sys.modules[module_name]

    else:
        build_extension = None
        if cython_inline.so_ext is None:
            # Figure out and cache current extension suffix
            build_extension = _get_build_extension()
            cython_inline.so_ext = build_extension.get_ext_filename('')

        lib_dir = os.path.abspath(lib_dir)
        module_path = os.path.join(lib_dir, module_name + cython_inline.so_ext)

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)
        if force or not os.path.isfile(module_path):
            define_macros = []
            c_include_dirs = []
            # import numpy
            # c_include_dirs.append(numpy.get_include())
            # define_macros.append(("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"))
            pyx_file = os.path.join(lib_dir, module_name + '.pyx')
            with open(pyx_file, 'w') as fh:
                fh.write(code)
            extension = Extension(
                name=module_name,
                sources=[pyx_file],
                include_dirs=c_include_dirs or None,
                define_macros=define_macros or None,
            )
            if build_extension is None:
                build_extension = _get_build_extension()
            build_extension.extensions = cythonize(
                [extension],
                include_path=cython_include_dirs or ['.'],
                compiler_directives=cython_compiler_directives,
                quiet=quiet,
                annotate=annotate,
            )
            build_extension.build_temp = os.path.dirname(pyx_file)
            build_extension.build_lib  = lib_dir
            build_extension.run()

        if sys.platform == 'win32' and sys.version_info >= (3, 8):
            with os.add_dll_directory(os.path.abspath(lib_dir)):
                module = load_dynamic(module_name, module_path)
        else:
            module = load_dynamic(module_name, module_path)

    return module
