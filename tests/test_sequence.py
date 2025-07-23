import pytest
import numpy as np
from array import array

from sdiff.sequence import diff, diff_nested
from sdiff.chunk import Diff, Chunk
from sdiff.numpy import get_backend_2d

from .util import np_chunk_eq


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_empty(kernel):
    assert diff("", "", kernel=kernel) == Diff(
        ratio=1,
        diffs=[],
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_equal_str(kernel):
    s = "alice bob"
    assert diff(s, s, kernel=kernel) == Diff(
        ratio=1,
        diffs=[
            Chunk(data_a=s, data_b=s, eq=True)
        ],
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_sub_str(kernel):
    assert diff("ice", "alice bob", kernel=kernel, min_ratio=0) == Diff(
        ratio=.5,
        diffs=[
            Chunk(data_a="", data_b="al", eq=False),
            Chunk(data_a="ice", data_b="ice", eq=True),
            Chunk(data_a="", data_b=" bob", eq=False),
        ],
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_sub_str_raw_codes(kernel):
    a = "ice"
    b = "alice bob"
    buffer = array('b', b'\xFF' * (len(a) + len(b)))
    assert diff(a, b, kernel=kernel, min_ratio=0, rtn_diff=buffer) == Diff(ratio=0.5, diffs=None)
    assert buffer == array("b", b"\x02\x02\x03\x00\x03\x00\x03\x00\x02\x02\x02\x02")


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_sub_str_early_stop_1(kernel):
    assert diff("xxx", "a xxx xx", kernel=kernel, min_ratio=0).ratio == 6 / 11


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_sub_str_early_stop_2(kernel):
    assert diff("xxx", "a xxx xx", kernel=kernel, min_ratio=0.9, strict=False) == Diff(
        ratio=4 / 11,
        diffs=[
            Chunk(data_a="x", data_b="a xxx ", eq=False),
            Chunk(data_a="xx", data_b="xx", eq=True),
        ],
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_sub_str_early_stop_3(kernel):
    assert diff("xxx", "a xxx xx", kernel=kernel, max_cost=2, strict=False) == Diff(
        ratio=4 / 11,
        diffs=[
            Chunk(data_a="x", data_b="a xxx ", eq=False),
            Chunk(data_a="xx", data_b="xx", eq=True),
        ],
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
@pytest.mark.parametrize("eq_only", [False, True])
def test_fuzz(kernel, eq_only):
    a = [5, 0, 3, 3, 7, 9, 3, 5, 2, 4, 7, 6, 8, 8, 1, 6, 7, 7, 8, 1, 5, 9,
         8, 9, 4, 3, 0, 3, 5, 0, 2, 3, 8, 1, 3, 3, 3, 7, 0, 1]
    b = [4, 1, 8, 5, 4, 3, 5, 7, 6, 6, 9, 3, 3, 2, 7, 3, 9, 9, 5, 9, 8, 3,
         8, 5, 9, 0, 7, 0, 1, 6, 6, 4, 5, 7, 6, 0, 1, 6, 6, 4]
    assert diff(a, b, kernel=kernel, min_ratio=0.425, eq_only=eq_only).ratio == 0.425


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_equal_str_nested_emulation(kernel):
    a, b = ["alice1", "bob1", "xxx"], ["alice2", "bob2"]

    def _eq(i: int, j: int):
        return diff(a[i], b[j], eq_only=True)

    def _dig(i: int, j: int):
        return diff(a[i], b[j])

    assert diff(a, b, kernel=kernel, eq=_eq, dig=_dig) == Diff(
        ratio=0.8,
        diffs=[
            Chunk(data_a=["alice1", "bob1"], data_b=["alice2", "bob2"], eq=[
                Diff(ratio=5 / 6, diffs=[
                    Chunk(data_a="alice", data_b="alice", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
                Diff(ratio=3 / 4, diffs=[
                    Chunk(data_a="bob", data_b="bob", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
            ]),
            Chunk(data_a=["xxx"], data_b=[], eq=False),
        ],
    )


def test_equal_str_nested_recursive():
    assert diff_nested(["alice1", "bob1", "xxx"], ["alice2", "bob2"]) == Diff(
        ratio=0.8,
        diffs=[
            Chunk(data_a=["alice1", "bob1"], data_b=["alice2", "bob2"], eq=[
                Diff(ratio=5 / 6, diffs=[
                    Chunk(data_a="alice", data_b="alice", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
                Diff(ratio=3 / 4, diffs=[
                    Chunk(data_a="bob", data_b="bob", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
            ]),
            Chunk(data_a=["xxx"], data_b=[], eq=False),
        ],
    )


def test_equal_str_nested_recursive_eq_only():
    assert diff_nested(["alice1", "bob1", "xxx"], ["alice2", "bob2"], rtn_diff=False) == Diff(ratio=0.8, diffs=None)
    assert diff_nested(["alice1", "bob1", "xxx"], ["alice2", "bob2"], eq_only=True) == Diff(ratio=0.8, diffs=None)


def test_complex_nested():
    a = ["alice1", "bob1", "xxx", [0, 1, 2, "charlie1"], [5, 6, 7]]
    b = ["alice2", "bob2", [0, 2, "charlie2"], [5, 8, 9]]

    assert diff_nested(a, b, min_ratio=0.5) == Diff(
        ratio=2 / 3,
        diffs=[
            Chunk(data_a=["alice1", "bob1"], data_b=["alice2", "bob2"], eq=[
                Diff(ratio=5 / 6, diffs=[
                    Chunk(data_a="alice", data_b="alice", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
                Diff(ratio=3 / 4, diffs=[
                    Chunk(data_a="bob", data_b="bob", eq=True),
                    Chunk(data_a="1", data_b="2", eq=False),
                ]),
            ]),
            Chunk(data_a=["xxx"], data_b=[], eq=False),
            Chunk(data_a=[[0, 1, 2, "charlie1"]], data_b=[[0, 2, "charlie2"]], eq=[
                Diff(
                    ratio=6 / 7,
                    diffs=[
                        Chunk(data_a=[0], data_b=[0], eq=True),
                        Chunk(data_a=[1], data_b=[], eq=False),
                        Chunk(data_a=[2], data_b=[2], eq=True),
                        Chunk(data_a=["charlie1"], data_b=["charlie2"], eq=[
                            Diff(
                                ratio=7 / 8,
                                diffs=[
                                    Chunk(data_a="charlie", data_b="charlie", eq=True),
                                    Chunk(data_a="1", data_b="2", eq=False),
                                ],
                            ),
                        ]),
                    ],
                ),
            ]),
            Chunk(data_a=[[5, 6, 7]], data_b=[[5, 8, 9]], eq=False),
        ],
    )


def test_complex_nested_raw():
    a = ["alice1", "bob1", "xxx", [0, 1, 2, "charlie1"], [5, 6, 7]]
    b = ["alice2", "bob2", [0, 2, "charlie2"], [5, 8, 9]]
    buffer = array('b', b'\xFF' * 9)
    assert diff_nested(a, b, min_ratio=0.5, rtn_diff=buffer) == Diff(ratio=2 / 3, diffs=None)
    assert buffer == array("b", b"\x03\x00\x03\x00\x01\x03\x00\x02\x01")


def test_nested_same():
    a = ["alice1", "bob1", "xxx", [0, 1, 2, "charlie1", []], [5, 6, 7]]
    assert diff_nested(a, a) == Diff(ratio=1.0, diffs=[Chunk(data_a=a, data_b=a, eq=True)])


def test_nested_cyclic():
    a = [[0, 1, 2], 0, 1, 2]
    a.append(a)
    b = [0, 1, 2, [0, 1, 2]]
    b.insert(0, b)

    with pytest.raises(ValueError, match="encountered recursive nesting of inputs"):
        diff_nested(a, b, min_ratio=0)


def test_nested_cost():
    a = [[0] * 10, [1] * 10, [2] * 10, [3] * 10]
    b = [[0] * 9 + [None], [1] * 8 + [None] * 2, [2] * 7 + [None] * 3, [3] * 6 + [None] * 4]
    assert diff_nested(a, b, min_ratio=(0.5, 0.8)) == Diff(
        ratio=0.5,
        diffs=[
            Chunk(
                data_a=a[:2],
                data_b=b[:2],
                eq=[
                    Diff(
                        ratio=0.9,
                        diffs=[
                            Chunk(data_a=[0] * 9, data_b=[0] * 9, eq=True),
                            Chunk(data_a=[0], data_b=[None], eq=False),
                        ],
                    ),
                    Diff(
                        ratio=0.8,
                        diffs=[
                            Chunk(data_a=[1] * 8, data_b=[1] * 8, eq=True),
                            Chunk(data_a=[1] * 2, data_b=[None] * 2, eq=False),
                        ],
                    ),
                ],
            ),
            Chunk(
                data_a=a[2:],
                data_b=b[2:],
                eq=False,
            )
        ]
    )


@pytest.mark.parametrize("max_depth", [10, 2])
def test_nested_np(monkeypatch, max_depth):
    a = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ])
    b = a.copy()
    b[1, 1] = 9
    b[3] += 100

    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    assert diff_nested(a, b, min_ratio=0.1) == Diff(
        ratio=5. / 6,
        diffs=[
            Chunk(data_a=a[:1], data_b=b[:1], eq=True),
            Chunk(data_a=a[1:2], data_b=b[1:2], eq=[
                Diff(ratio=2 / 3, diffs=[
                    Chunk(data_a=np.array([3]), data_b=np.array([3]), eq=True),
                    Chunk(data_a=np.array([4]), data_b=np.array([9]), eq=False),
                    Chunk(data_a=np.array([5]), data_b=np.array([5]), eq=True),
                ]),
            ]),
            Chunk(data_a=a[2:3], data_b=b[2:3], eq=True),
            Chunk(data_a=a[3:4], data_b=b[3:4], eq=False),
            Chunk(data_a=a[4:], data_b=b[4:], eq=True),
        ]
    )


@pytest.mark.parametrize("max_depth", [10, 2])
def test_big_np(monkeypatch, max_depth):
    np.random.seed(0)
    shape = (10, 1000)
    a = np.random.randint(0, 10, size=shape)
    b = np.random.randint(0, 10, size=shape)

    assert diff_nested(a, b, min_ratio=0, max_depth=max_depth).ratio > 0


def test_strictly_no_python_0():
    with pytest.raises(ValueError, match="failed to pick a type-aware protocol"):
        diff([0, 1, 2], [0, 1, 2], no_python=True)


def test_strictly_no_python_1():
    diff("abc", "abc", no_python=True)


def test_strictly_no_python_2():
    diff(b"abc", b"abc", no_python=True)


def test_strictly_no_python_3():
    diff(array("b", b"abc"), array("b", b"abc"), no_python=True)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32,
                                   np.float64, np.float128, np.object_, np.bool_, np.str_, np.bytes_])
def test_strictly_no_python_4(dtype):
    if dtype in (np.float16, np.str_, np.bytes_):
        pytest.skip("not implemented")
    a = np.arange(3).astype(dtype)
    diff(a, a, no_python=True)


def test_bug_0():
    a, b = 'comparing a.csv/b.csvX', 'comparing .X'
    assert diff(a, b, eq_only=True, min_ratio=0.75).ratio < 0.75


def test_bug_1():
    a, b = [1, 1, 0, 1, 1, 1, 1, 1, 1], [0, 0, 1]
    diff(a, b, min_ratio=0, kernel="py")


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_numpy_ext_2d(monkeypatch, kernel):
    a = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ])
    b = a.copy()
    b[1, 1] = 9
    b[3] += 100

    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    assert diff(a, b, eq=get_backend_2d(a, b), no_python=kernel == "c", accept=0.5, kernel=kernel) == Diff(
        ratio=5./6,
        diffs=[
            Chunk(a[:3], b[:3], eq=True),
            Chunk(a[3:4], b[3:4], eq=False),
            Chunk(a[4:], b[4:], eq=True),
        ]
    )


@pytest.mark.parametrize("kernel", ["py", "c"])
def test_numpy_ext_2d_weights(monkeypatch, kernel):
    a = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18],
    ])
    b = a.copy()
    b[1, 1] = 9
    b[3] += 100

    weights = array('d', [.5, 2, .5])

    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    assert diff(a, b, no_python=kernel == "c", eq=get_backend_2d(a, b, weights), accept=0.5,
                min_ratio=0, kernel=kernel) == Diff(
        ratio=2./3,
        diffs=[
            Chunk(a[:1], b[:1], eq=True),
            Chunk(a[1:2], b[1:2], eq=False),
            Chunk(a[2:3], b[2:3], eq=True),
            Chunk(a[3:4], b[3:4], eq=False),
            Chunk(a[4:], b[4:], eq=True),
        ]
    )


def test_empty_nested_0():
    assert diff_nested([[]], [[]]) == Diff(
        ratio=1.0,
        diffs=[Chunk(data_a=[[]], data_b=[[]], eq=True)],
    )