from typing import Optional, Callable, Any
from collections.abc import Sequence
from functools import reduce
from operator import mul

from .cython.tools import build_inline_module
from .cython.compare import ComparisonBackend
from .cython.struct3118 import parse_3118, Type, AtomicType, StructType

IMPORT = (
    "import cython",
    "from sdiff.cython.compare cimport ComparisonBackend",
)
CLASS_DEF = (
    "cdef class Backend(ComparisonBackend):",  # required to inherit from the base class
)
COMPARE_DEF = (
    "  @cython.initializedcheck(False)",
    "  @cython.wraparound(False)",
    "  cdef int compare(self, Py_ssize_t i, Py_ssize_t j):",
)
COMPARE_SIMPLE = (
    "    return self.a[i] == self.b[j]",
)
COMPARE_ABS = (
    "    cdef double delta = self.a[i] - self.b[j]",
    "    return (delta >= -self.atol) and (delta <= self.atol)",
)
RESOLVE_DEF = (
    "  @cython.initializedcheck(False)",
    "  @cython.wraparound(False)",
    "  def resolve(self, Py_ssize_t i, Py_ssize_t j):",
)


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


def wrap_python_callable(fun: Callable[[int, int], Any], resolver: Optional[Callable[[int, int], Any]], **kwargs) -> ComparisonBackend:
    """
    Wraps a python callable into a comparison protocol.

    The callable has to accept to integers (two positions in a, b sequences) and return
    an object that can be cast to a boolean indicating whether the elements can be aligned or not.

    This protocol is potentially slow as it will call ``fun`` from python runtime.

    Parameters
    ----------
    fun
        The callable comparing a pair of elements from a and b.
    resolver
        An optional accompanying callable providing detailed information (any object)
        on the comparison.
    kwargs
        Cython build arguments.

    Returns
    -------
    The comparison protocol for the callable.
    """
    init_args = {"callable": fun}
    fields = [("object", "callable")]
    if resolver is not None:
        init_args["resolver"] = resolver
        fields.append(("object", "resolver"))
    source_code = [
        *IMPORT,
        *CLASS_DEF,
        *compose_init(fields),
        *COMPARE_DEF,
        "    return bool(self.callable(i, j))",
    ]
    if resolver is not None:
        source_code.extend([
            *RESOLVE_DEF,
            "    return self.resolver(i, j)",
        ])
    return build_inline_module("\n".join(source_code), *kwargs).Backend(**init_args)


def wrap_python_pair(a: Sequence[Any], b: Sequence[Any], atol: Optional[float] = None, **kwargs) -> ComparisonBackend:
    """
    Wraps a pair of python sequences (or anything indexable) into a comparison protocol.

    This protocol is potentially slow as it will call ``a[i].__eq__(b[j])`` or ``a[i] - b[j]`` from python runtime.

    Parameters
    ----------
    a
    b
        The two sequences to compare.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    kwargs
        Cython build arguments.

    Returns
    -------
    The comparison protocol for the pair of python objects.
    """
    init_args = {"a": a, "b": b}
    fields = [
        ("object", "a"),
        ("object", "b"),
    ]
    if atol is not None:
        fields.append(("double", "atol"))
        init_args["atol"] = atol
    source_code = [
        *IMPORT,
        *CLASS_DEF,
        *compose_init(fields),
        *COMPARE_DEF,
        *(COMPARE_ABS if atol is not None else COMPARE_SIMPLE),
    ]
    return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)


def wrap_str(a: str, b: str, **kwargs) -> ComparisonBackend:
    """
    Wraps a pair of Unicode strings into a comparison protocol.

    Parameters
    ----------
    a
    b
        The two strings to compare.
    kwargs
        Cython build arguments.

    Returns
    -------
    The comparison protocol for the pair of strings.
    """
    init_args = {"a": a, "b": b}
    fields = [
        ("unicode", "a"),
        ("unicode", "b"),
    ]
    source_code = [
        *IMPORT,
        *CLASS_DEF,
        *compose_init(fields),
        *COMPARE_DEF,
        *COMPARE_SIMPLE,
    ]
    return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)


def wrap_memoryview(a: memoryview, b: memoryview, atol: Optional[float] = None, struct_threshold: Optional[int] = None,
                    struct_mask: Optional[Sequence[bool]] = None, **kwargs):
    """
    Wraps a pair of memory views into a comparison protocol.

    This wrapper is a high-performance workhorse to compare arrays of fixed types
    including ``bytes`` objects.

    Parameters
    ----------
    a
    b
        The two memory views to compare.
        They have to be one-dimensional and share the same item type.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    struct_threshold
        For arrays of struct data types (e.g. numpy record arrays) specifies
        the minimal number of fields to be equal for the whole struct to be
        considered equal.
    struct_mask
        For arrays of struct data types (e.g. numpy record arrays) specifies
        a mask including/removing individual fields into/from comparison.
        Masks for inner (nested) struct data types are not supported.
    kwargs
        Build arguments.

    Returns
    -------
    The resulting protocol.
    """
    if a.ndim != b.ndim:
        raise ValueError(f"tensors have different dimensionality: {a.ndim} != {b.ndim}")
    if a.ndim != 1:
        raise ValueError(f"unsupported dimensionality of tensors: {a.ndim}")

    struct_a = parse_3118(a.format)
    struct_b = parse_3118(b.format)

    if struct_a != struct_b:
        raise ValueError(f"item data types do not match: {struct_a} != {struct_b}")

    source_code = list(IMPORT)
    init_args = {"a": a, "b": b}
    _counter = 0
    _types = {}

    def _struct_def(_t: StructType, _name: str) -> list[str]:
        _code = [f"cdef packed struct {_name}:"]
        for _i, _field in enumerate(_t.fields):
            if _field.shape is None:
                _postfix = ""
            elif isinstance(_field.shape, int):
                _postfix = f"[{_field.shape}]"
            else:
                _postfix = f"[{','.join(map(str, _field.shape))}]"
            _code.append(f"  {_declare(_field.type)} f{_i}{_postfix}")
        return _code

    def _compare_expr(_left: str, _right: str, _type: Type) -> str:
        if isinstance(_type, AtomicType):
            if atol is not None and _type.typecode != 's':
                return  f"({_left} - {_right}) >= -atol and ({_left} - {_right}) <= atol"
            else:
                return f"{_left} == {_right}"
        elif isinstance(_type, StructType):
            _fields = [_left, _right]
            if atol is not None:
                _fields.append("atol")
            return f"compare_{_types[_type]}({', '.join(_fields)})"
        else:
            raise ValueError(f"unknown type: {_type}")

    def _declare(t: Type, mask: Optional[Sequence[bool]] = None, threshold: Optional[int] = None) -> str:
        """Declares a struct type and adds a comparison for it"""
        nonlocal _counter
        if isinstance(t, AtomicType):
            return t.c
        elif t in _types:
            return _types[t]
        elif isinstance(t, StructType):
            name = f"struct_{_counter}"
            _types[t] = name
            _counter += 1
            source_code.extend(_struct_def(t, name))
            _code = []
            _max_dims = 0
            _i = 0
            _need_temp = False
            for _i, _field in enumerate(t.fields):
                if mask is None or mask[_i]:
                    if _field.shape is None:
                        _shape = tuple()
                    elif isinstance(_field.shape, int):
                        _shape = (_field.shape,)
                    else:
                        _shape = _field.shape

                    _n_elements = 1
                    if _shape:
                        _n_elements = reduce(mul, _shape)
                    if _n_elements > 1:
                        _code.append(f"  temp = 0")
                        _need_temp = True

                    _indent = ""
                    _index = ""
                    for _j, _s in enumerate(_shape):
                        _code.append(f"  {_indent}for i{_j} in range({_s}):")
                        _indent += "  "
                        _index += f"[i{_j}]"
                    _max_dims = max(_max_dims, len(_shape))
                    _accumulator = "result" if _n_elements == 1 else "temp"
                    _code.append(f"  {_indent}{_accumulator} += {_compare_expr(f'a.f{_i}{_index}', f'b.f{_i}{_index}', _field.type)}")
                    if _accumulator == "temp":
                        _code.append(f"  result += temp == {_n_elements}")
            fields = f"const {name} a, const {name} b"
            if atol is not None:
                fields += ", const double atol"
            source_code.extend([
                f"cdef int compare_{name}({fields}):",
                f"  cdef int result = 0",
            ])
            if _need_temp:
                source_code.append("  cdef int temp = 0")
            if _max_dims:
                source_code.append(f"  cdef Py_ssize_t {', '.join(f'i{_j}' for _j in range(_max_dims))}")
            source_code.extend(_code)
            source_code.append(f"  return result >= {threshold or _i + 1}")
            return name
        else:
            raise ValueError(f"unknown type: {t}")

    if isinstance(struct_a, AtomicType) and struct_a.typecode == "O":
        dtype_str = "object"
        fields = [
            (f"{dtype_str}[:]", "a"),
            (f"{dtype_str}[:]", "b"),
        ]
    else:
        dtype_str = _declare(struct_a, mask=struct_mask, threshold=struct_threshold)
        fields = [
            (f"const {dtype_str}[:]", "a"),
            (f"const {dtype_str}[:]", "b"),
        ]

    if atol is not None:
        fields.append(("double", "atol"))
        init_args["atol"] = atol
    source_code.extend([
        *CLASS_DEF,
        *compose_init(fields),
        *COMPARE_DEF,
    ])

    if isinstance(struct_a, AtomicType):
        source_code.extend(COMPARE_ABS if atol is not None else COMPARE_SIMPLE)
    elif isinstance(struct_a, StructType):
        _args = "self.a[i], self.b[j]"
        if atol is not None:
            _args += ", self.atol"
        source_code.append(f"    return compare_{dtype_str}({_args})")
    else:
        raise ValueError(f"unknown type: {struct_a}")
    return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)


def wrap(arg, allow_python: bool = True, atol: Optional[float] = None, struct_threshold: Optional[int] = None,
         struct_mask: Optional[Sequence[bool]] = None, resolver: Optional[Callable[[int, int], Any]] = None, **kwargs):
    """
    Assembles the compare protocol from the provided argument.

    Parameters
    ----------
    arg
        The object to wrap into comparison backend. Can be
        - either a pair of indexable objects
        - or a callable(i, j)
    allow_python
        If set to True, will allow (slow) python kernels for
        comparing python lists, etc.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    struct_threshold
        For arrays of struct data types (e.g. numpy record arrays) specifies
        the minimal number of fields to be equal for the whole struct to be
        considered equal.
    struct_mask
        For arrays of struct data types (e.g. numpy record arrays) specifies
        a mask including/removing individual fields into/from comparison.
        Masks for inner (nested) struct data types are not supported.
    resolver
        An optional callable(i, j) comparing two elements in details.
        This callable will be used to populate the "details" field
        of the resulting diff.
    kwargs
        Build arguments.

    Returns
    -------
    The resulting protocol.
    """
    if isinstance(arg, ComparisonBackend):  # nothing to be done
        return arg

    if isinstance(arg, tuple):
        a, b = arg
        ta, tb = type(a), type(b)

        if ta == tb:
            if ta is str:
                return wrap_str(a, b, **kwargs)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except TypeError:
                    pass  # fallback to python comparison
                else:
                    return wrap_memoryview(a=mem_a, b=mem_b, atol=atol, struct_mask=struct_mask,
                                           struct_threshold=struct_threshold, **kwargs)
        if not allow_python:
            raise ValueError(f"failed to pick a type-aware protocol (failed to convert to memoryview or data type mismatch)")
        return wrap_python_pair(a=a, b=b, atol=atol, **kwargs)
    else:
        if not allow_python:
            raise ValueError(f"failed to pick a type-aware protocol (callable porovided)")
        return wrap_python_callable(fun=arg, resolver=resolver, **kwargs)
