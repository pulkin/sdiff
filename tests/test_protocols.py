import pytest
from array import array
import numpy as np

from sdiff.protocols import wrap
from sdiff.cython.struct3118 import parse_3118, StructType, StructField, AtomicType


def test_call_protocol():
    log = []

    def graph(*args):
        log.append(args)
        return 13

    comparison_backend = wrap(graph)
    assert comparison_backend(40, 42) == 13
    assert log == [(40, 42)]


def test_python_protocol():
    comparison_backend = wrap(([0, 1, 2], [2, 3]))
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(2, 0) == 1


def test_no_python_0():
    def graph(*args):
        return 1
    with pytest.raises(ValueError, match="failed to pick a type-aware protocol"):
        wrap(graph, allow_python=False)


def test_no_python_1():
    with pytest.raises(ValueError, match="failed to pick a type-aware protocol"):
        wrap(([0, 1, 2], [2, 3]), allow_python=False)


def test_str_protocol():
    comparison_backend = wrap(("abc", "cd"))
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(2, 0) == 1


@pytest.mark.parametrize("typecode", list("bBhHiIlLqQfd"))
def test_buffer_protocol(typecode):
    comparison_backend = wrap((
        array(typecode, [0, 1, 2]),
        array(typecode, [2, 3]),
    ))
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(2, 0) == 1


def array_typecode2format(typecode) -> str:
    return memoryview(array(typecode, [])).format


@pytest.mark.parametrize("typecode", list("bBhHiIlLqQfduw"))
def test_3118_array_simple_types(typecode):
    try:
        # typecode 'w' is not supported everywhere
        _format = array_typecode2format(typecode)
    except ValueError as e:
        pytest.skip(f"skipped {typecode} because of {e}")
    assert parse_3118(_format) == AtomicType(typecode="w" if typecode == "u" else typecode, byte_order="@")


def np_dtype2format(dtype) -> str:
    return memoryview(np.empty([], dtype=dtype)).format


@pytest.mark.parametrize("dtype, typecode", [
    (np.int8, 'b'),
    (np.uint8, 'B'),
    (np.int16, 'h'),
    (np.uint16, 'H'),
    (np.int32, 'i'),
    (np.uint32, 'I'),
    (np.int64, 'l'),
    (np.uint64, 'L'),
    (np.float16, 'e'),
    (np.float32, 'f'),
    (np.float64, 'd'),
    (np.float128, 'g'),
    (np.object_, 'O'),
    (np.bool_, '?'),
    (np.str_, 'w'),
    (np.bytes_, 's'),
    (np.complex64, 'Zf'),
    (np.complex128, 'Zd'),
])
def test_3118_np_simple_types(dtype, typecode):
    assert parse_3118(np_dtype2format(dtype)) == AtomicType(typecode=typecode[-1:], byte_order="@", z=typecode[0] == "Z")


@pytest.mark.parametrize("byte_order", ["<", ">", "="])
def test_3118_np_byte_order(byte_order):
    transformed = {"<": "@", ">": ">", "=": "@"}
    assert parse_3118(np_dtype2format(byte_order + "h")) == AtomicType(typecode="h", byte_order=transformed[byte_order])


@pytest.fixture
def nested_np_dtype():
    dtype_a = np.dtype([("ix", ">i8"), ("val", "<f", 2)])
    return np.dtype([('matrix', dtype_a, (3, 2)), ("weights", "f", (2, 3)), ('comment?_^', np.str_, 32)])


def test_3118_np_nested_struct(nested_np_dtype):
    assert parse_3118(np_dtype2format(nested_np_dtype)) == StructType((
        StructField(type=StructType((
            StructField(type=AtomicType(typecode="q", byte_order=">"), shape=None, caption="ix"),
            StructField(type=AtomicType(typecode="f", byte_order="@"), shape=(2,), caption="val"),
        )), shape=(3, 2), caption="matrix"),
        StructField(type=AtomicType(typecode="f", byte_order="@"), shape=(2,3), caption="weights"),
        StructField(type=AtomicType(typecode="w", byte_order="@"), shape=32, caption="comment?_^"),
    ))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
                                   np.float64, np.float128, np.object_, np.bool_, np.str_, np.bytes_])
def test_buffer_protocol_np(dtype):
    if dtype in (np.float16, np.str_):
        pytest.skip("not supported in cython")
    comparison_backend = wrap((
        np.array([0, 1, 2], dtype=dtype),
        np.array([2, 3], dtype=dtype),
    ))
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(2, 0) == 1


def test_np_record():
    dtype = np.dtype([("ix", "i8"), ("val", "f", 2)])
    comparison_backend = wrap((
        np.array([(0, (3.14, 159)), (1, (2.71, 828)), (2, (6.02, 333))], dtype=dtype),
        np.array([(2, (6.02, 333)), (0, (3.14, 36.0))], dtype=dtype),
    ))
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(0, 1) == 2./3
    assert comparison_backend(2, 0) == 1
