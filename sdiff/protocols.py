from typing import Optional

from .cython.tools import build_inline_module
from .cython.compare import ComparisonBackend

PREAMBLE = (
    "import cython",
    "from sdiff.cython.compare cimport ComparisonBackend",
    "cdef class Backend(ComparisonBackend):",  # required to inherit from the base class
)
COMPARE_DEF = (
    "  @cython.initializedcheck(False)",
    "  @cython.wraparound(False)",
    "  cdef double compare(self, Py_ssize_t i, Py_ssize_t j):",
)
COMPARE_SIMPLE = (
    "    return self.a[i] == self.b[j]",
)
COMPARE_ABS = (
    "    cdef double delta = self.a[i] - self.b[j]",
    "    return (delta >= -self.e_abs) and (delta <= self.e_abs)",
)
dtypes_conversion = {
    "c": "const char",
    "b": "const signed char",
    "B": "const unsigned char",
    "h": "const short",
    "H": "const unsigned short",
    "i": "const int",
    "I": "const unsigned int",
    "l": "const long",
    "L": "const unsigned long",
    "q": "const long long",
    "Q": "const unsigned long long",
    "f": "const float",
    "d": "const double",
    "g": "const long double",
    "?": "const unsigned char",
    "O": "object",
}


def compose_init(fields: list[tuple[str, str]]) -> list[str]:
    def _autofmt(t, n):
        result = f"{t} {n}"
        if '[:' in t:
            result += ' not None'
        return result

    source_code = []
    for t, n in fields:
        source_code.append(f"  cdef {t} {n}")
    source_code.append(f"  def __init__(self, {', '.join(_autofmt(t, n) for t, n in fields)}):")
    for _, n in fields:
        source_code.append(f"    self.{n} = {n}")
    return source_code


def wrap(data, allow_python: bool = True, e_abs: Optional[float] = None, **kwargs):
    """
    Figures out the compare protocol from the argument.

    Parameters
    ----------
    data
        The object to wrap into comparison backend. Can be
        - either a pair of array-like objects
        - or a callable(i, j)
    allow_python
        If set to True, will allow (slow) python kernels for
        comparing python lists, etc.
    e_abs
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= e_abs``
        instead of the equality comparison ``a[i] == b[j]``.
    kwargs
        Build arguments.

    Returns
    -------
    The resulting protocol.
    """
    if isinstance(data, ComparisonBackend):  # nothing to be done
        return data
    source_code = list(PREAMBLE)

    is_pair = False

    if isinstance(data, tuple):
        a, b = data
        ta, tb = type(a), type(b)
        is_pair = True
        init_args = {"a": a, "b": b}

        if ta == tb:
            if ta is str:
                if e_abs is not None:
                    raise ValueError("cannot use e_abs for str comparison")
                source_code.extend(compose_init([
                    ("unicode", "a"),
                    ("unicode", "b"),
                ]))
                source_code.extend(COMPARE_DEF)
                source_code.extend(COMPARE_SIMPLE)
                return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except TypeError:
                    pass
                else:
                    if mem_a.ndim != mem_b.ndim:
                        raise ValueError(f"tensors have different dimensionality: {mem_a.ndim} != {mem_b.ndim}")
                    dtype_str_a = dtypes_conversion[mem_a.format]
                    dtype_str_b = dtypes_conversion[mem_b.format]
                    if mem_a.ndim == 1:
                        _vars = [
                            (dtype_str_a + "[:]", "a"),
                            (dtype_str_b + "[:]", "b"),
                        ]
                        if e_abs is not None:
                            _vars.append(("double", "e_abs"))
                            init_args["e_abs"] = e_abs
                        source_code.extend(compose_init(_vars))
                        source_code.extend(COMPARE_DEF)
                        source_code.extend(COMPARE_ABS if e_abs is not None else COMPARE_SIMPLE)
                        return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)
                    else:
                        raise ValueError(f"unsupported dimensionality of tensors: {mem_a.ndim}")

    if not allow_python:
        raise ValueError("failed to pick a type-aware protocol")
    if is_pair:
        _vars = [
            ("object", "a"),
            ("object", "b"),
        ]
        if e_abs is not None:
            _vars.append(("double", "e_abs"))
            init_args["e_abs"] = e_abs
        source_code.extend(compose_init(_vars))
        source_code.extend(COMPARE_DEF)
        source_code.extend(COMPARE_ABS if e_abs is not None else COMPARE_SIMPLE)
        return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)
    else:
        if e_abs is not None:
            raise ValueError("cannot use e_abs for callables")
        source_code.extend(compose_init([
            ("object", "callable"),
        ]))
        source_code.extend(COMPARE_DEF)
        source_code.append(
            "    return self.callable(i, j)",
        )
        return build_inline_module("\n".join(source_code), *kwargs).Backend(data)
