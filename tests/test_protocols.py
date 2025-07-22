import pytest
from array import array
import numpy as np

from sdiff.protocols import wrap


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


def test_buffer_protocol_2d():
    comparison_backend = wrap((
        np.array([[0, .5], [1, 1.5], [2, 2.5]]),
        np.array([[2, 2.5], [3, 3.5]]),
    ), allow_k2d=True)
    assert comparison_backend(0, 0) == 0
    assert comparison_backend(2, 0) == 1
