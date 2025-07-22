import array
from random import randint, seed
import numpy as np

import pytest

from sdiff.myers import search_graph_recursive
from sdiff.cython.cmyers import search_graph_recursive as csearch_graph_recursive
from sdiff.cython.tools import build_inline_module
from sdiff.protocols import wrap
from sdiff.sequence import canonize

#TODO: refactor
ComparisonCallBackend = type(wrap(None))
ComparisonStrBackend = type(wrap(("abc", "def")))
MapBackend = build_inline_module(
    """
    from sdiff.cython.compare cimport ComparisonBackend
    cdef class Backend(ComparisonBackend):
      cdef const unsigned char[:, :] map
      def __init__(self, const unsigned char[:, :] map):
        self.map = map
      cdef double compare(self, Py_ssize_t i, Py_ssize_t j):
        return self.map[i, j]
    """
).Backend


def compute_cost(codes):
    return sum(i % 3 != 0 for i in codes)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("n, m", [(1, 1), (2, 2), (7, 4), (7, 7)])
@pytest.mark.parametrize("v", [0, 1])
def test_empty_full(driver, n, m, v):
    def graph(i: int, j: int) -> float:
        return v

    result = array.array('b', b'\xFF' * (n + m))
    cost = driver(n, m, ComparisonCallBackend(graph), result)
    assert compute_cost(result) == cost
    if v == 0:
        assert cost == n + m
        canonize(result)
        assert list(result) == [1] * n + [2] * m
    else:
        assert cost == abs(n - m)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_quantized_1(driver):
    def graph(i: int, j: int) -> float:
        return i == 2 * j

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, ComparisonCallBackend(graph), result)
    assert compute_cost(result) == cost
    assert cost == 3
    canonize(result)
    assert list(result) == [3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0]


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_dummy_1(driver):
    def graph(i: int, j: int) -> float:
        return i == j and i % 2

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, ComparisonCallBackend(graph), result)
    assert compute_cost(result) == cost
    assert cost == 7
    canonize(result)
    assert list(result) == [1, 2, 3, 0, 1, 2, 3, 0, 1, 1, 1]


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_dummy_2(driver):
    def graph(i: int, j: int) -> float:
        return i == j and i % 2

    result = array.array('b', b'\xFF' * 11)
    cost = driver(4, 7, ComparisonCallBackend(graph), result)
    assert compute_cost(result) == cost
    assert cost == 7
    canonize(result)
    assert list(result) == [1, 2, 3, 0, 1, 2, 3, 0, 2, 2, 2]


@pytest.mark.parametrize("max_cost, expected_cost, expected", [
    (2, 7, [3, 0] + [1] * 5 + [2] * 2 + [3, 0]),
    (3, 3.0, [3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0]),  # 3 is the the breakpoint for this case
])
@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_max_cost_quantized(driver, max_cost, expected_cost, expected):
    def graph(i: int, j: int) -> float:
        return i == 2 * j

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, ComparisonCallBackend(graph), result, max_cost=max_cost)
    assert compute_cost(result) == cost
    assert cost == expected_cost
    canonize(result)
    assert list(result) == expected


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_eq_only(driver):
    result = array.array('b', b'\xFF' * 18)
    with pytest.warns(UserWarning, match="the 'out' argument is ignored for eq_only=True"):
        cost = driver(9, 9, ComparisonStrBackend("aaabbbccc", "aaaxxxccc"), result, eq_only=True, max_cost=8)
    assert cost == 6


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_str_non_periodic(driver):
    result = array.array('b', b'\xFF' * 18)
    cost = driver(9, 9, ComparisonStrBackend("aaabbbccc", "dddbbbeee"), result)
    assert compute_cost(result) == cost
    assert cost == 12
    canonize(result)
    assert list(result) == [1] * 3 + [2] * 3 + [3, 0] * 3 + [1] * 3 + [2] * 3


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_str_non_periodic_2(driver):
    result = array.array('b', b'\xFF' * 18)
    cost = driver(9, 9, ComparisonStrBackend("aaabbbccc", "aaadddccc"), result)
    assert compute_cost(result) == cost
    assert cost == 6
    canonize(result)
    assert list(result) == [3, 0] * 3 + [1] * 3 + [2] * 3 + [3, 0] * 3


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_max_calls(driver):
    a = "_0aaa1_"
    b = "_2aaa3_"
    assert driver(len(a), len(b), ComparisonStrBackend(a, b)) == 4
    assert driver(len(a), len(b), ComparisonStrBackend(a, b), max_calls=2) == 10


def check_codes_valid(eq, codes, n, m):
    iter_codes = iter(codes)
    x = y = 0
    for code in iter_codes:
        if code == 1:
            x += 1
        elif code == 2:
            y += 1
        elif code == 3:
            assert next(iter_codes) == 0, "code 3 is not followed by code 0"
            q = eq(x, y)
            assert bool(q), f"eq({x}, {y}) = {q} does not eval to True"
            x += 1
            y += 1
        else:
            raise AssertionError(f"unknown {code=}")
    assert x == n, f"{x=} != {n=}"
    assert y == m, f"{y=} != {m=}"


def check_code_valid_seq(a, b, codes):
    check_codes_valid(lambda i, j: a[i] == b[j], codes, len(a), len(b))


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("rtn_diff", [False, True])
def test_fuzz(driver, rtn_diff):
    for s in range(400):
        seed(s)
        n = randint(10, 100)
        m = randint(10, 100)
        f = randint(0, 100)
        _map = np.random.randint(0, 99, size=(n, m)) < f

        if rtn_diff:
            result = array.array('b', b'\xFF' * (n + m))
        else:
            result = None
        cost = driver(n, m, MapBackend(_map), result)
        if rtn_diff:
            assert compute_cost(result) == cost
            check_codes_valid(lambda i, j: _map[i, j], result, n, m)
