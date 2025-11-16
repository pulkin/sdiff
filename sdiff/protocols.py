from typing import Optional, Callable, Any
from collections.abc import Sequence
from array import array
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


def wrap(arg, allow_python: bool = True, atol: Optional[float] = None, struct_weights: Optional[array] = None,
         struct_threshold: float = 0.749, resolver: Optional[Callable[[int, int], Any]] = None, **kwargs):
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
    struct_weights
        Optional weights for structure type comparison.
    struct_threshold
        The threshold for equal structs.
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
    source_code = list(IMPORT)

    is_pair = False

    if isinstance(arg, tuple):
        a, b = arg
        ta, tb = type(a), type(b)
        is_pair = True
        init_args = {"a": a, "b": b}

        if ta == tb:
            if ta is str:
                return wrap_str(a, b, **kwargs)
            else:
                try:
                    mem_a = memoryview(a)
                    mem_b = memoryview(b)
                except TypeError:
                    pass
                else:

                    if mem_a.ndim != mem_b.ndim:
                        raise ValueError(f"tensors have different dimensionality: {mem_a.ndim} != {mem_b.ndim}")

                    struct_a = parse_3118(mem_a.format)
                    struct_b = parse_3118(mem_b.format)

                    if struct_a != struct_b:
                        raise ValueError(f"types do not match: {struct_a} != {struct_b}")

                    _counter = 0
                    _types = {}
                    def _declare(t: Type, array_exact: bool = True, do_weights: bool = False) -> str:
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
                            _code = [f"cdef packed struct {name}:"]
                            for _i, _field in enumerate(t.fields):
                                if _field.shape is None:
                                    _postfix = ""
                                elif isinstance(_field.shape, int):
                                    _postfix = f"[{_field.shape}]"
                                else:
                                    _postfix = f"[{','.join(map(str, _field.shape))}]"
                                _code.append(f"  {_declare(_field.type)} f{_i}{_postfix}")
                            source_code.extend(_code)
                            _code = []
                            _max_dims = 0
                            _need_temp = False
                            for _i, _field in enumerate(t.fields):
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
                                if _n_elements == 1:
                                    _accumulator = "result"
                                else:
                                    _accumulator = "temp"

                                if isinstance(_field.type, AtomicType):
                                    _eq = f"a.f{_i}{_index} == b.f{_i}{_index}"
                                    if atol is not None and _field.type.typecode != 's':
                                        _eq = f"(a.f{_i}{_index} - b.f{_i}{_index}) >= -atol and (a.f{_i}{_index} - b.f{_i}{_index}) <= atol"
                                    if do_weights and _accumulator == "result":
                                        _eq = f"({_eq}) * weights[{_i}]"
                                    _code.append(f"  {_indent}{_accumulator} += {_eq}")
                                elif isinstance(_field.type, StructType):
                                    _args = f"a.f{_i}{_index}, b.f{_i}{_index}, threshold"
                                    if atol is not None:
                                        _args += ", atol"
                                    _code.append(f"  {_indent}{_accumulator} += compare_{_types[_field.type]}({_args})")
                                else:
                                    raise ValueError(f"unknown type: {_field.type}")
                                if _accumulator == "temp":
                                    if array_exact:
                                        _eq = f"temp == {_n_elements}"
                                    else:
                                        _eq = f"temp / {_n_elements}"
                                    if do_weights:
                                        _eq = f"({_eq}) * weights[{_i}]"
                                    _code.append(f"  result += {_eq}")
                            _args = f"const {name} a, const {name} b, const double threshold"
                            if do_weights:
                                _args += ", const double[:] weights"
                            if atol is not None:
                                _args += ", const double atol"
                            source_code.extend([
                                f"cdef int compare_{name}({_args}):",
                                f"  cdef double result = 0",
                            ])
                            if _need_temp:
                                source_code.append("  cdef double temp = 0")
                            if _max_dims:
                                source_code.append(f"  cdef Py_ssize_t {', '.join(f'i{_j}' for _j in range(_max_dims))}")
                            source_code.extend(_code)
                            if len(t.fields) == 1:
                                _postfix = ""
                            else:
                                _postfix = f" / {len(t.fields)}"
                            source_code.append(f"  return result{_postfix} >= threshold")
                            return name
                        else:
                            raise ValueError(f"unknown type: {t}")

                    if isinstance(struct_a, AtomicType) and struct_a.typecode == "O":
                        dtype_str = "object"
                        _vars = [
                            (f"{dtype_str}[:]", "a"),
                            (f"{dtype_str}[:]", "b"),
                        ]
                    else:
                        dtype_str = _declare(struct_a, do_weights=True)
                        _vars = [
                            (f"const {dtype_str}[:]", "a"),
                            (f"const {dtype_str}[:]", "b"),
                        ]
                        if isinstance(struct_a, StructType):
                            _vars.extend([
                                ("double", "threshold"),
                                ("const double[:]", "weights"),
                            ])

                    if mem_a.ndim == 1:
                        if atol is not None:
                            _vars.append(("double", "atol"))
                            init_args["atol"] = atol
                        source_code.extend(CLASS_DEF)
                        source_code.extend(compose_init(_vars))
                        source_code.extend(COMPARE_DEF)

                        if isinstance(struct_a, AtomicType):
                            source_code.extend(COMPARE_ABS if atol is not None else COMPARE_SIMPLE)
                        elif isinstance(struct_a, StructType):
                            _args = "self.a[i], self.b[j], self.threshold, self.weights"
                            if atol is not None:
                                _args += ", self.atol"
                            if struct_weights is None:
                                struct_weights = array('d', [1.] * len(struct_a.fields))
                            init_args["threshold"] = struct_threshold
                            init_args["weights"] = array('d', struct_weights)
                            source_code.append(f"    return compare_{dtype_str}({_args})")
                        else:
                            raise ValueError(f"unknown type: {struct_a}")
                        return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)
                    else:
                        raise ValueError(f"unsupported dimensionality of tensors: {mem_a.ndim}")

    if not allow_python:
        raise ValueError("failed to pick a type-aware protocol")
    if is_pair:
        a, b = arg
        return wrap_python_pair(a=a, b=b, atol=atol, **kwargs)
    else:
        return wrap_python_callable(fun=arg, resolver=resolver, **kwargs)
