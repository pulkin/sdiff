import array
from random import choice, randint, seed

import pytest

from sdiff.myers import search_graph_recursive
from sdiff.cython.cmyers import search_graph_recursive as csearch_graph_recursive
from sdiff.sequence import canonize


def compute_cost(codes):
    return sum(i % 3 != 0 for i in codes)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("n, m", [(1, 1), (2, 2), (7, 4), (7, 7)])
def test_empty(driver, n, m):
    def complicated_graph(i: int, j: int) -> float:
        return 0

    result = array.array('b', b'\xFF' * (n + m))
    cost = driver(n, m, complicated_graph, result)
    assert compute_cost(result) == cost
    assert cost == n + m
    canonize(result)
    assert list(result) == [1] * n + [2] * m


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_quantized_1(driver):
    def complicated_graph(i: int, j: int) -> float:
        return i == 2 * j

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, complicated_graph, result)
    assert compute_cost(result) == cost
    assert cost == 3
    canonize(result)
    assert list(result) == [3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0]


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_dummy_1(driver):
    def complicated_graph(i: int, j: int) -> float:
        return i == j and i % 2

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, complicated_graph, result)
    assert compute_cost(result) == cost
    assert cost == 7
    canonize(result)
    assert list(result) == [1, 2, 3, 0, 1, 2, 3, 0, 1, 1, 1]


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_impl_dummy_2(driver):
    def complicated_graph(i: int, j: int) -> float:
        return i == j and i % 2

    result = array.array('b', b'\xFF' * 11)
    cost = driver(4, 7, complicated_graph, result)
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
    def complicated_graph(i: int, j: int) -> float:
        return i == 2 * j

    result = array.array('b', b'\xFF' * 11)
    cost = driver(7, 4, complicated_graph, result, max_cost=max_cost)
    assert compute_cost(result) == cost
    assert cost == expected_cost
    canonize(result)
    assert list(result) == expected


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_eq_only(driver):
    result = array.array('b', b'\xFF' * 18)
    with pytest.warns(UserWarning, match="the 'out' argument is ignored for eq_only=True"):
        cost = driver(9, 9, ("aaabbbccc", "aaaxxxccc"), result, eq_only=True, max_cost=8)
    assert cost == 6


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_str_non_periodic(driver):
    result = array.array('b', b'\xFF' * 18)
    cost = driver(9, 9, ("aaabbbccc", "dddbbbeee"), result)
    assert compute_cost(result) == cost
    assert cost == 12
    canonize(result)
    assert list(result) == [1] * 3 + [2] * 3 + [3, 0] * 3 + [1] * 3 + [2] * 3


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_str_non_periodic_2(driver):
    result = array.array('b', b'\xFF' * 18)
    cost = driver(9, 9, ("aaabbbccc", "aaadddccc"), result)
    assert compute_cost(result) == cost
    assert cost == 6
    canonize(result)
    assert list(result) == [3, 0] * 3 + [1] * 3 + [2] * 3 + [3, 0] * 3


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
def test_max_calls(driver):
    a = "_0aaa1_"
    b = "_2aaa3_"
    assert driver(len(a), len(b), (a, b)) == 4
    assert driver(len(a), len(b), (a, b), max_calls=2) == 10


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
@pytest.mark.parametrize("s", list(range(100)))
@pytest.mark.parametrize("rtn_diff", [False, True])
def test_fuzz_call_long_short(driver, s, rtn_diff):
    seed(s)
    n = randint(10, 100)
    m = randint(10, 100)
    f = randint(0, 100)
    _map = {(x, y): randint(0, 99) < f for x in range(n) for y in range(m)}

    def compare(i, j):
        return _map[i, j]

    if rtn_diff:
        result = array.array('b', b'\xFF' * (n + m))
    else:
        result = None
    cost = driver(n, m, compare, result)
    if rtn_diff:
        assert compute_cost(result) == cost
        check_codes_valid(compare, result, n, m)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("s", list(range(100, 200)))
@pytest.mark.parametrize("rtn_diff", [False, True])
def test_fuzz_str_long_short(driver, s, rtn_diff):
    seed(s)
    n = randint(10, 100)
    m = randint(10, 100)
    f = randint(0, 100)
    choices = "a" * (100 - f) + "c" * f
    a = ''.join(choice(choices) for _ in range(n))
    b = ''.join(choice(choices) for _ in range(m))

    if rtn_diff:
        result = array.array('b', b'\xFF' * (n + m))
    else:
        result = None
    cost = driver(n, m, (a, b), result)
    if rtn_diff:
        assert compute_cost(result) == cost
        check_code_valid_seq(a, b, result)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("s", list(range(200, 300)))
@pytest.mark.parametrize("rtn_diff", [False, True])
def test_fuzz_array_long_short(driver, s, rtn_diff):
    seed(s)
    n = randint(10, 100)
    m = randint(10, 100)
    f = randint(0, 100)
    choices = "a" * (100 - f) + "c" * f
    a = array.array('q', [choice(choices) for _ in range(n)])
    b = array.array('q', [choice(choices) for _ in range(m)])

    if rtn_diff:
        result = array.array('b', b'\xFF' * (n + m))
    else:
        result = None
    cost = driver(n, m, (a, b), result)
    if rtn_diff:
        assert compute_cost(result) == cost
        check_code_valid_seq(a, b, result)


@pytest.mark.parametrize("driver", [search_graph_recursive, csearch_graph_recursive])
@pytest.mark.parametrize("s", list(range(300, 400)))
@pytest.mark.parametrize("rtn_diff", [False, True])
def test_fuzz_list_long_short(driver, s, rtn_diff):
    seed(s)
    n = randint(10, 100)
    m = randint(10, 100)
    f = randint(0, 100)
    choices = "a" * (100 - f) + "c" * f
    a = [choice(choices) for _ in range(n)]
    b = [choice(choices) for _ in range(m)]

    if rtn_diff:
        result = array.array('b', b'\xFF' * (n + m))
    else:
        result = None
    cost = driver(n, m, (a, b), result)
    if rtn_diff:
        assert compute_cost(result) == cost
        check_code_valid_seq(a, b, result)
