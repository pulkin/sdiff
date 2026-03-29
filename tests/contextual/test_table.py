import pytest
import numpy as np

from sdiff.contextual.table import diff, TableDiff
from sdiff.numpy import NumpyDiff
from sdiff.chunk import Signature, ChunkSignature

from ..util import np_raw_diff_eq


@pytest.fixture
def a():
    np.random.seed(0)
    return np.random.randint(0, 10, size=(10, 10))


@pytest.fixture
def a1(a):
    return a + np.eye(10, dtype=a.dtype)


def test_equal_auto(monkeypatch, a):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff(a, a, "table") == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=a,
            eq=np.ones_like(a, dtype=bool),
            row_diff_sig=Signature.aligned(10),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )


def test_equal_numeric_dtype(monkeypatch, a):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff(a, a, "table", dtype=a.dtype, fill=0) == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=a,
            eq=np.ones_like(a, dtype=bool),
            row_diff_sig=Signature.aligned(10),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )


def test_aligned(monkeypatch, a, a1):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff(a, a1, "table", fill=0) == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=a1,
            eq=(a == a1),
            row_diff_sig=Signature.aligned(10),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )


def test_missing_0(monkeypatch, a):
    """Missing row in head"""
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    b = a.copy()
    b[0] = 0
    eq = np.ones_like(a, dtype=bool)
    eq[0] = False

    assert diff(a, a[1:], "table", fill=0) == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=b,
            eq=eq,
            row_diff_sig=Signature(
                (ChunkSignature.delta(1, 0), ChunkSignature.aligned(9))
            ),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )


def test_missing_1(monkeypatch, a):
    """Missing row in tail"""
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    b = a.copy()
    b[-1] = 0
    eq = np.ones_like(a, dtype=bool)
    eq[-1] = False

    assert diff(a, a[:-1], "table", fill=0) == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=b,
            eq=eq,
            row_diff_sig=Signature(
                (ChunkSignature.aligned(9), ChunkSignature.delta(1, 0))
            ),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )


def test_missing_2(monkeypatch, a):
    """Missing row in the middle"""
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    b = a.copy()
    b[3] = 0
    eq = np.ones_like(a, dtype=bool)
    eq[3] = False

    assert diff(a, np.concat([a[:3], a[4:]], axis=0), "table", fill=0) == TableDiff(
        name="table",
        data=NumpyDiff(
            a=a,
            b=b,
            eq=eq,
            row_diff_sig=Signature(
                (
                    ChunkSignature.aligned(3),
                    ChunkSignature.delta(1, 0),
                    ChunkSignature.aligned(6),
                )
            ),
            col_diff_sig=Signature.aligned(10),
        ),
        columns=None,
    )
