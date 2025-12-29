from typing import Optional, Callable, Any
from collections.abc import Generator, Sequence
from dataclasses import dataclass

from .cython.tools import build_inline_module
from .cython.compare import ComparisonBackend
from .cython.struct3118 import parse_3118, Type, AtomicType, StructType, StructField

IMPORT = (
    "import cython",
    "from cpython cimport array",
    "import array",
    "from sdiff.cython.compare cimport ComparisonBackend",
    "cdef array.array _int_array_template = array.array('i', [])"
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

    The callable has to accept two integers (two positions in a, b sequences) and return
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


@dataclass(frozen=True)
class _CompareSpec:
    left: StructType
    right: StructType
    field_map: tuple[tuple[str, str], ...]

    def __post_init__(self):
        errors = []
        if delta_a := ({i for i, _ in self.field_map} - set(self.left.fields_by_name)):
            errors.append(f"missing fields {', '.join(map(str, delta_a))} in the first data type")
        if delta_b := ({j for _, j in self.field_map} - set(self.right.fields_by_name)):
            errors.append(f"missing fields {', '.join(map(str, delta_b))} in the second data type")
        if errors:
            raise ValueError("; ".join(errors))

    @classmethod
    def from_pair(cls, left: StructType, right: StructType) -> "_CompareSpec":
        if (n := len(left.fields)) != (m := len(right.fields)):
            raise ValueError(f"structs have different field count: {n} != {m}")
        field_map = tuple(
            (fa.caption, fb.caption)
            for fa, fb in zip(left.fields, right.fields)
        )
        return _CompareSpec(left, right, field_map)

    def walk(self, _visited: Optional[set["_CompareSpec"]] = None) -> Generator["_CompareSpec", None, None]:
        if _visited is None:
            _visited = set()
        if self in _visited:
            return
        yield self

        _visited.add(self)
        for left_field, right_field in self.field_map:
            left_type = self.left.fields[self.left.fields_by_name[left_field]].type
            right_type = self.right.fields[self.right.fields_by_name[right_field]].type
            if isinstance(left_type, StructType) and isinstance(right_type, StructType):
                yield from _CompareSpec.from_pair(left_type, right_type).walk(_visited=_visited)


class _MVCodeGen:
    """A class for cython code generation for memoryviews"""
    def __init__(self, atol: Optional[float] = None):
        self.atol = atol

    def get_type_c_name(self, t: Type) -> str:
        """Picks a name for the provided type"""
        if isinstance(t, AtomicType):
            return t.c
        elif isinstance(t, StructType):
            return f"struct_{t.get_fingerprint()[:16]}"
        else:
            raise NotImplementedError(f"unknown type: {t}")

    def get_struct_compare_def_name(self, left_t: StructType, right_t: StructType, prefix: str = "compare") -> str:
        """A naming convention for structure comparison"""
        return f"{prefix}_{self.get_type_c_name(left_t)}_{self.get_type_c_name(right_t)}"

    def get_struct_cdef(self, t: StructType) -> list[str]:
        """Prepares a cdef section for the provided struct type"""
        code = [f"cdef packed struct {self.get_type_c_name(t)}:"]
        for i, field in enumerate(t.fields):
            if field.shape is None:
                postfix = ""
            elif isinstance(field.shape, int):
                postfix = f"[{field.shape}]"
            else:
                postfix = f"[{']['.join(map(str, field.shape))}]"
            code.append(f"  const {self.get_type_c_name(field.type)} field{i}{postfix}")
        return code

    def get_type_compare_expr(self, left: str, left_t: Type, right: str, right_t: Type) -> str:
        """Prepares a comparison expression for the provided type"""
        if isinstance(left_t, AtomicType) and isinstance(right_t, AtomicType):
            if self.atol is not None and left_t.typecode != 's' and right_t.typecode != 's':
                return  f"({left} - {right}) >= -atol and ({left} - {right}) <= atol"
            elif left_t.typecode == 's' != right_t.typecode == 's':  # prevent string-number comparison
                raise ValueError("attempt to compare strings and numbers")
            else:
                return f"{left} == {right}"

        elif isinstance(left_t, StructType) and isinstance(right_t, StructType):
            fields = [left, right]
            if self.atol is not None:
                fields.append("atol")
            return f"{self.get_struct_compare_def_name(left_t, right_t)}({', '.join(fields)})"
        else:
            raise ValueError(f"unknown or incompatible types: {left_t}, {right_t}")

    def get_struct_field_comparison(
            self,
            left: str,
            left_field: StructField,
            right: str,
            right_field: StructField,
            result_statement: str,
            indent="",
    ) -> list[str]:
        if left_field.shape != right_field.shape:
            raise ValueError(f"cannot compare fields with differing shapes: {left_field} vs {right_field}")
        shape = left_field.shape

        if shape is None:
            return [f"{indent}{result_statement.format(expr=self.get_type_compare_expr(left, left_field.type, right, right_field.type))}"]
        elif isinstance(shape, int):
            shape = (shape,)

        code = []

        subscript = ""
        for i, s in enumerate(shape):
            code.append(f"{indent}for i{i} in range({s}):")
            indent += "  "
            subscript += f"[i{i}]"

        code.extend([
            f"{indent}if not ({self.get_type_compare_expr(f'{left}{subscript}', left_field.type, f'{right}{subscript}', right_field.type)}):",
            f"{indent}  break",
        ])

        for i, s in reversed(list(enumerate(shape))):
            indent = indent[:-2]
            if i == 0:
                code.extend([
                    f"{indent}else:",
                    f"{indent}  {result_statement.format(expr='1')}",
                ])
            else:
                code.extend([
                    f"{indent}else:",
                    f"{indent}  continue",
                    f"{indent}break",
                ])
        return code

    def get_struct_compare_code(
            self,
            left_t: StructType,
            right_t: StructType,
            field_map: Sequence[tuple[str, str]],
            threshold: bool = False,
    ) -> list[str]:
        """Declares a struct type and adds a comparison for it"""
        left_n = self.get_type_c_name(left_t)
        right_n = self.get_type_c_name(right_t)
        code = []

        args = f"const {left_n} a, const {right_n} b"
        if threshold:
            args += ", const long threshold"
        if self.atol is not None:
            args += ", const double atol"

        code.extend([
            f"cdef int {self.get_struct_compare_def_name(left_t, right_t)}({args}):",
            f"  cdef int result = 0",
        ])
        for left_name, right_name in field_map:
            left_ix = left_t.fields_by_name[left_name]
            right_ix = right_t.fields_by_name[right_name]
            left_field = left_t.fields[left_ix]
            right_field = right_t.fields[right_ix]
            code.extend(self.get_struct_field_comparison(
                f"a.field{left_ix}",
                left_field,
                f"b.field{right_ix}",
                right_field,
                "result += {expr}",
                indent="  ",
            ))
        code.append(f"  return result >= {'threshold' if threshold else len(field_map)}")
        return code

    def get_resolve_code(
            self,
            left_t: StructType,
            right_t: StructType,
            field_map: Sequence[tuple[str, str]],
    ) -> list[str]:
        left_n = self.get_type_c_name(left_t)
        right_n = self.get_type_c_name(right_t)
        code = []

        args = f"const {left_n} a, const {right_n} b"
        if self.atol is not None:
            args += ", const double atol"

        code.extend([
            f"@cython.wraparound(False)",
            f"@cython.boundscheck(False)",
            f"cdef {self.get_struct_compare_def_name(left_t, right_t, 'resolve')}({args}):",
            f"  cdef array.array result = array.clone(_int_array_template, {len(field_map)}, zero=True)",
            f"  cdef int[:] result_view = result"
        ])
        for i, (left_name, right_name) in enumerate(field_map):
            left_ix = left_t.fields_by_name[left_name]
            right_ix = right_t.fields_by_name[right_name]
            left_field = left_t.fields[left_ix]
            right_field = right_t.fields[right_ix]
            code.extend(self.get_struct_field_comparison(
                f"a.field{left_ix}",
                left_field,
                f"b.field{right_ix}",
                right_field,
                f"result_view[{i}] = {{expr}}",
                indent="  ",
            ))
        code.append(f"  return result")
        return code


def wrap_memoryview(a: memoryview, b: memoryview, atol: Optional[float] = None, struct_threshold: Optional[int] = None,
                    struct_field_map: Optional[Sequence[tuple[str, str]]] = None, **kwargs):
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
    struct_field_map
        For arrays of struct data types (e.g. numpy record arrays) accepts
        a map indicating which pairs of fields have to be compared. Each element
        is a pair of field names in the two structures. There is currently no
        way to specify the map for inner (nested) structures.
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

    if isinstance(struct_a, StructType) and isinstance(struct_b, StructType):
        structs = True
    elif isinstance(struct_a, AtomicType) and isinstance(struct_b, AtomicType):
        structs = False
    else:
        raise ValueError(f"item data types do not match: {struct_a} != {struct_b}")

    source_code = list(IMPORT)
    fields = []
    init_args = {"a": a, "b": b}
    codegen = _MVCodeGen(atol=atol)

    for side, name in zip((struct_a, struct_b), "ab"):
        if isinstance(side, AtomicType) and side.typecode == "O":
            fields.append((f"object[:]", name))
        else:
            fields.append((f"const {codegen.get_type_c_name(side)}[:]", name))

    if structs:
        root_comparison = _CompareSpec.from_pair(struct_a, struct_b) if struct_field_map is None else _CompareSpec(struct_a, struct_b, tuple(struct_field_map))
        structs_declared = set()
        for comparison in list(root_comparison.walk())[::-1]:
            for side in comparison.left, comparison.right:
                if side not in structs_declared:
                    source_code.extend(codegen.get_struct_cdef(side))
                    structs_declared.add(side)
            source_code.extend(codegen.get_struct_compare_code(comparison.left, comparison.right, comparison.field_map, threshold=struct_threshold is not None and comparison is root_comparison))
        source_code.extend(codegen.get_resolve_code(root_comparison.left, root_comparison.right, root_comparison.field_map))
        if struct_threshold is not None:
            fields.append((f"long", "threshold"))
            init_args["threshold"] = struct_threshold

    if atol is not None:
        fields.append(("double", "atol"))
        init_args["atol"] = atol
    source_code.extend([
        *CLASS_DEF,
        *compose_init(fields),
        *COMPARE_DEF,
    ])

    if not structs:
        source_code.extend(COMPARE_ABS if atol is not None else COMPARE_SIMPLE)
    else:
        _args = ""
        if struct_threshold is not None:
            _args += ", self.threshold"
        if atol is not None:
            _args += ", self.atol"
        source_code.append(f"    return {codegen.get_struct_compare_def_name(struct_a, struct_b)}(self.a[i], self.b[j]{_args})")
        _args = ""
        if atol is not None:
            _args += ", self.atol"
        source_code.extend([
            *RESOLVE_DEF,
            f"    return {codegen.get_struct_compare_def_name(struct_a, struct_b, 'resolve')}(self.a[i], self.b[j]{_args})",
        ])
    return build_inline_module("\n".join(source_code), **kwargs).Backend(**init_args)


def wrap(arg, allow_python: bool = True, atol: Optional[float] = None, struct_threshold: Optional[int] = None,
         struct_field_map: Optional[Sequence[tuple[str, str]]] = None, resolver: Optional[Callable[[int, int], Any]] = None, **kwargs):
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
    struct_field_map
        For arrays of struct data types (e.g. numpy record arrays) accepts
        a map indicating which pairs of fields have to be compared. Each element
        is a pair of field names in the two structures. There is currently no
        way to specify the map for inner (nested) structures.
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
                    return wrap_memoryview(a=mem_a, b=mem_b, atol=atol, struct_threshold=struct_threshold,
                                           struct_field_map=struct_field_map, **kwargs)
        if not allow_python:
            raise ValueError(f"failed to pick a type-aware protocol (failed to convert to memoryview or data type mismatch)")
        return wrap_python_pair(a=a, b=b, atol=atol, **kwargs)
    else:
        if not allow_python:
            raise ValueError(f"failed to pick a type-aware protocol (callable porovided)")
        return wrap_python_callable(fun=arg, resolver=resolver, **kwargs)
