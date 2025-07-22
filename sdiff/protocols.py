import array

from .cython.tools import build_inline_module

dtypes = {
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


def wrap(data, allow_python: bool = True, allow_k2d: bool = False, k2d_weights=None):
    """
    Figures out the compare protocol from the argument.

    Parameters
    ----------
    data
        The object to wrap into comparison backend. Can be
        - either a pair of array-like objects
        - or a python callable(i, j)
    allow_python
        If set to True, will allow (slow) python kernels for
        comparing python lists, etc.
    allow_k2d
        If set to True, will allow 2D numpy array kernel.
    k2d_weights
        Optional weights for the previous.

    Returns
    -------
    The resulting protocol.
    """
    source_code = [
        "from sdiff.cython.compare cimport ComparisonBackend",
        "cdef class Backend(ComparisonBackend):",  # required to inherit from the base class
    ]
    simple_compare = [
        "  cdef double compare(self, Py_ssize_t i, Py_ssize_t j):",
        "    return self.a[i] == self.b[j]",
    ]

    def _add_fields(fields: list[tuple[str, str]]):
        for t, n in fields:
            source_code.append(f"  cdef {t} {n}")
        source_code.append(f"  def __init__(self, {', '.join(t + ' ' + n for t, n in fields)}):")
        for _, n in fields:
            source_code.append(f"    self.{n} = {n}")

    is_pair = False

    if isinstance(data, tuple):
        a, b = data
        ta, tb = type(a), type(b)
        is_pair = True

        if ta == tb:
            if ta is str:
                _add_fields([
                    ("unicode", "a"),
                    ("unicode", "b"),
                ])
                source_code.extend(simple_compare)
                return build_inline_module("\n".join(source_code)).Backend(a, b)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except TypeError:
                    pass
                else:
                    if mem_a.ndim != mem_b.ndim:
                        raise ValueError(f"tensors have different dimensionality: {mem_a.ndim} != {mem_b.ndim}")
                    dtype_str_a = dtypes[mem_a.format]
                    dtype_str_b = dtypes[mem_b.format]
                    if mem_a.ndim == 1:
                        _add_fields([
                            (dtype_str_a + "[:]", "a"),
                            (dtype_str_b + "[:]", "b"),
                        ])
                        source_code.extend(simple_compare)
                        return build_inline_module("\n".join(source_code)).Backend(a, b)
                    elif mem_a.ndim == 2 and allow_k2d:
                        if mem_a.shape[1] != mem_b.shape[1]:
                            raise ValueError(f"mismatch of the trailing dimension for 2D extension: {mem_a.shape[1]} != {mem_b.shape[1]}")
                        if k2d_weights is None:
                            k2d_weights = array.array('d', [1] * mem_a.shape[1])
                        _add_fields([
                            (dtype_str_a + "[:, :]", "a"),
                            (dtype_str_b + "[:, :]", "b"),
                            ("const double[:]", "weights"),
                        ])
                        source_code.extend([
                            "    assert a.shape[1] == b.shape[1]",
                            "    assert a.shape[1] == weights.shape[0]",
                            "  cdef double compare(self, Py_ssize_t i, Py_ssize_t j):",
                            "    cdef:",
                            "      Py_ssize_t t",
                            "      double result = 0",
                            "    for t in range(self.weights.shape[0]):",
                            "      result += (self.a[i, t] == self.b[j, t]) * self.weights[t]",
                            "    return result / self.weights.shape[0]",
                        ])
                        return build_inline_module("\n".join(source_code)).Backend(a, b, k2d_weights)
                    else:
                        raise ValueError(f"unsupported dimensionality of tensors: {mem_a.ndim}")

    if not allow_python:
        raise ValueError("failed to pick a type-aware protocol")
    if is_pair:
        _add_fields([
            ("object", "a"),
            ("object", "b"),
        ])
        source_code.extend(simple_compare)
        return build_inline_module("\n".join(source_code)).Backend(a, b)
    else:
        _add_fields([("object", "callable")])
        source_code.extend([
            "  cdef double compare(self, Py_ssize_t i, Py_ssize_t j):",
            "    return self.callable(i, j)",
        ])
        return build_inline_module("\n".join(source_code)).Backend(data)
