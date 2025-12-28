from array import array

import numpy as np
import pytest

from sdiff.chunk import Diff, Chunk, Signature, ChunkSignature
from sdiff.numpy import (diff, get_row_col_diff, align_inflate, diff_aligned_2d, NumpyDiff, dtype_diff,
                         align_inflate_arrays)

from .util import np_chunk_eq, np_raw_diff_eq, np_chunk_eq_numpy_details


@pytest.fixture
def a():
    np.random.seed(0)
    return np.random.randint(0, 10, size=(10, 10))


@pytest.fixture
def a1(a):
    return a + np.eye(10, dtype=a.dtype)


def test_equal(monkeypatch, a):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    assert diff(a, a) == Diff(
        ratio=1,
        diffs=[
            Chunk(data_a=a, data_b=a, eq=True, details=[
                Diff(ratio=1, diffs=[Chunk(data_a=row, data_b=row, eq=True)])
                for row in a
            ])
        ]
    )


def test_random(monkeypatch, a):
    b = a.copy()
    b[1:, 1] = 11
    b[2:, 2] = 12
    b[3] = 13

    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    assert diff(a, b) == Diff(
        ratio=0.9,
        diffs=[
            Chunk(data_a=a[:3], data_b=b[:3], eq=True, details=[
                Diff(ratio=1.0, diffs=[
                    Chunk(data_a=a[0], data_b=b[0], eq=True),
                ]),
                Diff(ratio=0.9, diffs=[
                    Chunk(data_a=a[1, :1], data_b=b[1, :1], eq=True),
                    Chunk(data_a=a[1, 1:2], data_b=b[1, 1:2], eq=False),
                    Chunk(data_a=a[1, 2:], data_b=b[1, 2:], eq=True),
                ]),
                Diff(ratio=0.8, diffs=[
                    Chunk(data_a=a[2, :1], data_b=b[2, :1], eq=True),
                    Chunk(data_a=a[2, 1:3], data_b=b[2, 1:3], eq=False),
                    Chunk(data_a=a[2, 3:], data_b=b[2, 3:], eq=True),
                ]),
            ]),
            Chunk(data_a=a[3:4], data_b=b[3:4], eq=False),
            Chunk(data_a=a[4:], data_b=b[4:], eq=True, details=[
                Diff(ratio=0.8, diffs=[
                    Chunk(data_a=_a[:1], data_b=_b[:1], eq=True),
                    Chunk(data_a=_a[1:3], data_b=_b[1:3], eq=False),
                    Chunk(data_a=_a[3:], data_b=_b[3:], eq=True),
                ])
                for _a, _b in zip(a[4:], b[4:])
            ])
        ]
    )


def test_row_col_sig_eq_0(a):
    assert get_row_col_diff(a, a) == (
        Signature(parts=(ChunkSignature(10, 10, True),)),
        Signature(parts=(ChunkSignature(10, 10, True),)),
    )


def test_row_col_sig_eq_1(a, a1):
    assert get_row_col_diff(a, a1) == (
        Signature(parts=(ChunkSignature(10, 10, True),)),
        Signature(parts=(ChunkSignature(10, 10, True),)),
    )


def test_row_col_sig_row(a):
    b = a.copy()
    b[4] += 1
    assert get_row_col_diff(a, b) == (
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
        Signature(parts=(
            ChunkSignature(10, 10, True),
        )),
    )


def test_row_col_sig_row_atol_0(a):
    b = a.copy()
    b[4] += 1
    assert get_row_col_diff(a, b, atol=0.5) == (
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
        Signature(parts=(
            ChunkSignature(10, 10, True),
        )),
    )


def test_row_col_sig_row_atol_1(a):
    b = a.copy()
    b[4] += 1
    assert get_row_col_diff(a, b, atol=1.5) == (
        Signature(parts=(ChunkSignature(10, 10, True),)),
        Signature(parts=(ChunkSignature(10, 10, True),)),
    )


def test_row_col_sig_col(a):
    b = a.copy()
    b[:, 4] += 1
    assert get_row_col_diff(a, b) == (
        Signature(parts=(
            ChunkSignature(10, 10, True),
        )),
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
    )


def test_row_col_sig_col_atol_0(a):
    b = a.copy()
    b[:, 4] += 1
    assert get_row_col_diff(a, b, atol=0.5) == (
        Signature(parts=(
            ChunkSignature(10, 10, True),
        )),
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
    )


def test_row_col_sig_col_atol_1(a):
    b = a.copy()
    b[:, 4] += 1
    assert get_row_col_diff(a, b, atol=1.5) == (
        Signature(parts=(ChunkSignature(10, 10, True),)),
        Signature(parts=(ChunkSignature(10, 10, True),)),
    )


def test_row_col_sig_row_col(a):
    b = a.copy()
    b[4] += 1
    b[:, 4] += 1
    assert get_row_col_diff(a, b) == (
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
        Signature(parts=(
            ChunkSignature(4, 4, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(5, 5, True),
        )),
    )


def test_align_inflate():
    a = np.arange(5)
    b = np.arange(5, 11)
    s = Signature(parts=[
        ChunkSignature(size_a=1, size_b=1, eq=True),
        ChunkSignature(size_a=2, size_b=3, eq=False),
        ChunkSignature(size_a=2, size_b=2, eq=True),
    ])
    a_, b_ = align_inflate(a, b, -1, s, 0)
    assert (a_ == np.array([0, 1, 2, -1, -1, -1, 3, 4])).all()
    assert (b_ == np.array([5, -1, -1, 6, 7, 8, 9, 10])).all()


@pytest.mark.parametrize("col_diff_sig", [None, Signature(parts=(ChunkSignature(10, 10, True),))])
def test_diff_aligned_2d_same_0(monkeypatch, a, a1, col_diff_sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff_aligned_2d(a, a1, 0, col_diff_sig=col_diff_sig) == NumpyDiff(
        a=a,
        b=a1,
        eq=(a == a1),
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=Signature.aligned(10),
    )


def test_diff_aligned_2d_same_1(monkeypatch, a):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    at = np.insert(a, 4, 0, axis=1)
    bt = np.insert(a, 3, 0, axis=1)
    mask = at == bt
    mask[:, 3:5] = False

    assert diff_aligned_2d(
        a, a, 0,
        col_diff_sig=(col_diff_sig := Signature(parts=(
            ChunkSignature(3, 3, True),
            ChunkSignature(1, 1, False),
            ChunkSignature(6, 6, True),
        )))
    ) == NumpyDiff(
        a=at,
        b=bt,
        eq=mask,
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=col_diff_sig,
    )


@pytest.mark.parametrize("col_diff_sig", [None, Signature(parts=(ChunkSignature(10, 10, True),))])
def test_diff_aligned_2d_atol_0(monkeypatch, a, a1, col_diff_sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff_aligned_2d(a, a1, 0, col_diff_sig=col_diff_sig, atol=0.5) == NumpyDiff(
        a=a,
        b=a1,
        eq=(a == a1),
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=Signature.aligned(10),
    )


@pytest.mark.parametrize("col_diff_sig", [None, Signature(parts=(ChunkSignature(10, 10, True),))])
def test_diff_aligned_2d_atol_1(monkeypatch, a, a1, col_diff_sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    assert diff_aligned_2d(a, a1, 0, col_diff_sig=col_diff_sig, atol=1.5) == NumpyDiff(
        a=a,
        b=a1,
        eq=np.ones_like(a, dtype=bool),
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=Signature.aligned(10),
    )


def test_diff_aligned_2d_new_row(monkeypatch, a, a1):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    at = np.insert(a, 4, 0, axis=0)
    bt = np.insert(a1, 4, 0, axis=0)
    mask = at == bt
    mask[4, :] = False

    assert diff_aligned_2d(a, bt, 0) == NumpyDiff(
        a=at,
        b=bt,
        eq=mask,
        row_diff_sig=Signature((
            ChunkSignature.aligned(4),
            ChunkSignature.delta(0, 1),
            ChunkSignature.aligned(6),
        )),
        col_diff_sig=Signature.aligned(10),
    )


def test_diff_aligned_2d_new_col(monkeypatch, a, a1):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    at = np.insert(a, 4, 0, axis=1)
    bt = np.insert(a1, 4, 0, axis=1)
    mask = at == bt
    mask[:, 4] = False

    assert diff_aligned_2d(a, bt, 0) == NumpyDiff(
        a=at,
        b=bt,
        eq=mask,
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=Signature((
            ChunkSignature.aligned(4),
            ChunkSignature.delta(0, 1),
            ChunkSignature.aligned(6),
        )),
    )


def test_diff_aligned_2d_new_row_col(monkeypatch, a, a1):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    at = np.insert(np.insert(a, 4, 0, axis=0), 8, 0, axis=1)
    bt = np.insert(np.insert(a1, 4, 0, axis=0), 8, 0, axis=1)
    mask = at == bt
    mask[4, :] = mask[:, 8] = False

    assert diff_aligned_2d(a, bt, 0) == NumpyDiff(
        a=at,
        b=bt,
        eq=mask,
        row_diff_sig=Signature((
            ChunkSignature.aligned(4),
            ChunkSignature.delta(0, 1),
            ChunkSignature.aligned(6),
        )),
        col_diff_sig=Signature((
            ChunkSignature.aligned(8),
            ChunkSignature.delta(0, 1),
            ChunkSignature.aligned(2),
        )),
    )


def test_diff_aligned_2d_mix_0(monkeypatch, a, a1):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    a = np.insert(np.insert(a, 4, 42, axis=0), 8, 42, axis=1)
    a1 = np.insert(np.insert(a1, 4, 89, axis=0), 8, 89, axis=1)

    at = np.insert(np.insert(a, 5, 0, axis=0), 9, 0, axis=1)
    bt = np.insert(np.insert(a1, 4, 0, axis=0), 8, 0, axis=1)
    mask = at == bt
    mask[4:6, :] = mask[:, 8:10] = False

    assert diff_aligned_2d(a, a1, 0) == NumpyDiff(
        a=at,
        b=bt,
        eq=mask,
        row_diff_sig=Signature((
            ChunkSignature.aligned(4),
            ChunkSignature.delta(1, 1),
            ChunkSignature.aligned(6),
        )),
        col_diff_sig=Signature((
            ChunkSignature.aligned(8),
            ChunkSignature.delta(1, 1),
            ChunkSignature.aligned(2),
        )),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.object_])
def test_dtype(monkeypatch, a, a1, dtype):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    a = a.astype(dtype)
    a1 = a1.astype(dtype)

    assert diff_aligned_2d(a, a1, 0) == NumpyDiff(
        a=a,
        b=a1,
        eq=(a == a1),
        row_diff_sig=Signature.aligned(10),
        col_diff_sig=Signature.aligned(10),
    )


def test_to_plain(monkeypatch, a, a1):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq_numpy_details)

    row_sig = Signature((
        ChunkSignature(3, 3, True),
        ChunkSignature(2, 1, False),
        ChunkSignature(2, 2, True),
        ChunkSignature(0, 2, False),
    ))
    col_sig = Signature((ChunkSignature(10, 10, True),))
    eq = a == a1
    eq[:2] = True

    diff = NumpyDiff(
        a=a,
        b=a1,
        eq=eq,  # this is slightly incorrect for non-equal rows
        row_diff_sig=row_sig,
        col_diff_sig=col_sig,
    )

    assert diff.to_plain() == Diff(
        ratio=2/3,
        diffs=[
            Chunk(data_a=a[:2], data_b=a1[:2], eq=True),
            Chunk(data_a=a[2:3], data_b=a1[2:3], eq=True, details=eq[2:3]),
            Chunk(data_a=a[3:5], data_b=a1[5:6], eq=False),
            Chunk(data_a=a[6:8], data_b=a1[6:8], eq=True, details=eq[6:8]),
            Chunk(data_a=a[8:8], data_b=a1[8:10], eq=False),
        ]
    )


@pytest.mark.parametrize("sig", [None, Signature.aligned(0)])
def test_empty_col_0(monkeypatch, sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    e = np.empty(shape=(42, 0))

    assert diff_aligned_2d(e, e, 0, col_diff_sig=sig) == NumpyDiff(
        a=e,
        b=e,
        eq=e.astype(bool),
        row_diff_sig=Signature.aligned(42),
        col_diff_sig=Signature.aligned(0),
    )


@pytest.mark.parametrize("sig", [None, Signature.aligned(0)])
def test_empty_col_0(monkeypatch, sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    e1 = np.empty(shape=(40, 0))
    e2 = np.empty(shape=(42, 0))

    assert diff_aligned_2d(e1, e2, 0, col_diff_sig=sig) == NumpyDiff(
        a=e2,
        b=e2,
        eq=e2.astype(bool),
        row_diff_sig=Signature((ChunkSignature(40, 40, True), ChunkSignature(0, 2, False),)),
        col_diff_sig=Signature.aligned(0),
    )


@pytest.mark.parametrize("sig", [None, Signature((ChunkSignature(10, 0, False),))])
def test_empty_col_2(monkeypatch, a, sig):
    monkeypatch.setattr(NumpyDiff, "__eq__", np_raw_diff_eq)

    e = a[:, :0]
    a_ = np.concat([a, np.zeros_like(a)], axis=0)

    assert diff_aligned_2d(a, e, 0, col_diff_sig=sig) == NumpyDiff(
        a=a_,
        b=np.zeros_like(a_),
        eq=np.zeros_like(a_, dtype=bool),
        row_diff_sig=Signature((ChunkSignature(10, 10, False),)),
        col_diff_sig=Signature((ChunkSignature(10, 0, False),)),
    )


def test_dtype_diff_0():
    i8 = np.dtype("i8")
    f8 = np.dtype("f8")
    f4 = np.dtype("f4")
    s32 = np.dtype("S32")

    d1 = np.dtype([("ix", i8), ("value_1", f8), ("value_2", f8), ("comment", s32)])
    d2 = np.dtype([("ix", i8), ("value_3", f8), ("value_2", f4)])

    assert dtype_diff(d1, d2, min_ratio=0, look_field_names=False, look_field_dtypes=True) == Diff(
        ratio=4./7,
        diffs=[
            Chunk(data_a=["ix", "value_1"], data_b=["ix", "value_3"], eq=True),
            Chunk(data_a=["value_2", "comment"], data_b=["value_2"], eq=False),
        ]
    )


def test_dtype_diff_1():
    i8 = np.dtype("i8")
    f8 = np.dtype("f8")
    f4 = np.dtype("f4")
    s32 = np.dtype("S32")

    d1 = np.dtype([("ix", i8), ("value_1", f8), ("value_2", f8), ("comment", s32)])
    d2 = np.dtype([("ix", i8), ("value_3", f8), ("value_2", f4)])

    assert dtype_diff(d1, d2, look_field_names=True, look_field_dtypes=True, min_ratio=0) == Diff(
        ratio=2./7,
        diffs=[
            Chunk(data_a=["ix"], data_b=["ix"], eq=True),
            Chunk(data_a=["value_1", "value_2", "comment"], data_b=["value_3", "value_2"], eq=False),
        ]
    )


def test_dtype_diff_2():
    i8 = np.dtype("i8")
    s32 = np.dtype("S32")

    d1 = np.dtype([("a1", i8), ("a2", i8), ("a3", i8), ("string", s32), ("a4", i8)])
    d2 = np.dtype([("b1", i8), ("string", s32), ("b2", i8), ("b3", i8), ("b4", i8)])

    _a = np.arange(11)
    _b = np.arange(1, 10)

    data_a = np.rec.fromarrays([_a, _a + 10, _a + 20, ["s"] * 11, _a + 30], dtype=d1)
    data_b = np.rec.fromarrays([_b + 10, ["x"] * 9, _b + 30, _b + 90, _b + 80], dtype=d2)

    assert dtype_diff(data_a, data_b, min_ratio=0, look_field_names=False, look_field_dtypes=True, look_field_data=True) == Diff(
        ratio=0.4,
        diffs=[
            Chunk(data_a=["a1"], data_b=[], eq=False),
            Chunk(data_a=["a2"], data_b=["b1"], eq=True),
            Chunk(data_a=["a3", "string"], data_b=["string"], eq=False),
            Chunk(data_a=["a4", ], data_b=["b2"], eq=True),
            Chunk(data_a=[], data_b=["b3", "b4"], eq=False),
        ]
    )


def test_dtype_diff_atomic_0():
    i8 = np.dtype("i8")

    a = np.arange(12)
    b = np.arange(1, 11)

    assert dtype_diff(a, b, look_field_data=True) == Diff(ratio=1, diffs=[
        Chunk(data_a=["field"], data_b=["field"], eq=True),
    ])


def test_dtype_diff_atomic_1():
    i8 = np.dtype("i8")

    a = np.arange(12)
    b = np.arange(1, 11)

    assert dtype_diff(a, b, look_field_data=True, data_min_ratio=0.95) == Diff(ratio=0, diffs=[
        Chunk(data_a=["field"], data_b=["field"], eq=False),
    ])


def test_align_inflate_arrays():
    i8 = np.dtype("i8")
    s32 = np.dtype("S32")

    d1 = np.dtype([("a1", i8), ("a2", i8), ("a3", i8), ("x1", s32), ("a4", i8)])
    d2 = np.dtype([("b1", i8), ("y1", s32), ("b2", i8), ("b3", i8), ("b4", i8)])

    _a = np.ones(shape=(3, 4), dtype=i8)
    _as = np.full(fill_value="a", shape=(3, 4), dtype=s32)
    _b = np.ones(shape=(5,), dtype=i8)
    _bs = np.full(fill_value="b", shape=(5,), dtype=s32)

    data_a = np.rec.fromarrays([_a, _a, _a, _as, _a], dtype=d1)
    data_b = np.rec.fromarrays([_b, _bs, _b, _b, _b], dtype=d2)

    d = dtype_diff(data_a, data_b, look_field_names=False, look_field_dtypes=True)
    assert d.ratio == 0.8

    data_a_aligned, data_b_aligned = align_inflate_arrays(data_a, data_b, d)

    assert (data_a_aligned == np.rec.fromarrays(
        [_a, np.zeros(_a.shape, dtype=s32), _a, _a, _as, _a],
        dtype=np.dtype([("a1", i8), ("y1", s32), ("a2", i8), ("a3", i8), ("x1", s32), ("a4", i8)]),
    )).all()
    assert (data_b_aligned == np.rec.fromarrays(
        [_b, _bs, _b, _b, np.zeros(_b.shape, dtype=s32), _b],
        dtype=np.dtype([("b1", i8), ("y1", s32), ("b2", i8), ("b3", i8), ("x1", s32), ("b4", i8)]),
    )).all()


def test_diff_record_zeros(monkeypatch):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    i8 = np.dtype("i8")
    s32 = np.dtype("S32")

    d1 = np.dtype([("a1", i8), ("a2", i8), ("a3", i8), ("x1", s32), ("a4", i8)])
    d2 = np.dtype([("b1", i8), ("y1", s32), ("b2", i8), ("b3", i8), ("b4", i8)])

    a = np.zeros(10, dtype=d1)
    b = np.zeros(10, dtype=d2)

    assert diff(a, b) == Diff(
        ratio=1,
        diffs=[Chunk(data_a=a, data_b=b, eq=True, details=[array('i', [1, 1, 1, 1])] * 10)],
    )


def test_diff_record_0(monkeypatch):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    i8 = np.dtype("i8")
    s32 = np.dtype("S32")

    d1 = np.dtype([("a1", i8), ("a2", i8), ("a3", i8), ("x1", s32), ("a4", i8)])
    d2 = np.dtype([("b1", i8), ("y1", s32), ("b2", i8), ("b3", i8), ("b4", i8)])

    a = np.rec.fromarrays([
        np.full(shape=10, fill_value=1, dtype=i8),
        np.full(shape=10, fill_value=2, dtype=i8),
        np.full(shape=10, fill_value=3, dtype=i8),
        np.zeros(shape=10, dtype=s32),
        np.full(shape=10, fill_value=4, dtype=i8),
    ], dtype=d1)
    b = np.rec.fromarrays([
        np.full(shape=10, fill_value=5, dtype=i8),
        np.zeros(shape=10, dtype=s32),
        np.full(shape=10, fill_value=6, dtype=i8),
        np.full(shape=10, fill_value=7, dtype=i8),
        np.full(shape=10, fill_value=8, dtype=i8),
    ], dtype=d2)

    assert diff(a, b, min_ratio=0.01) == Diff(
        ratio=0,
        diffs=[Chunk(data_a=a, data_b=b, eq=False)],
    )


def test_diff_record_1(monkeypatch):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    i8 = np.dtype("i8")
    s32 = np.dtype("S32")

    d1 = np.dtype([("a1", i8), ("a2", i8), ("a3", i8), ("x1", s32), ("a4", i8)])
    d2 = np.dtype([("b1", i8), ("y1", s32), ("b2", i8), ("b3", i8), ("b4", i8)])

    a = np.zeros(shape=10, dtype=d1)
    b = np.zeros(shape=10, dtype=d2)

    assert diff(a, b, record_min_ratio=0.81) == Diff(
        ratio=0,
        diffs=[Chunk(data_a=np.zeros(shape=10, dtype=d1), data_b=np.zeros(shape=10, dtype=d2), eq=False)],
    )
    assert diff(a, b, record_min_ratio=0.79) == Diff(
        ratio=1,
        diffs=[Chunk(
            data_a=np.zeros(shape=10, dtype=d1),
            data_b=np.zeros(shape=10, dtype=d2),
            eq=True,
            details=[array('i', [1, 1, 1, 1])] * 10,
        )],
    )


def test_diff_record_2(monkeypatch):
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq)

    i8 = np.dtype("i8")
    s32 = np.dtype("S32")
    f8 = np.dtype("f8")

    d1 = np.dtype([("ix", i8), ("name", s32), ("val", f8)])
    d2 = np.dtype([("ix", i8), ("name", s32), ("val_2", f8)])

    a = np.rec.fromarrays([
        np.array([0, 1, 2, 3, 4, 5], dtype=i8),
        np.array([b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta"], dtype=s32),
        np.array([10.2, 34.7, 12, 1, 7.7, 12.0], dtype=f8)
    ], dtype=d1)
    b = np.rec.fromarrays([
        np.array([0, 1, 2, 3, 4, 5], dtype=i8),
        np.array([b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta"], dtype=s32),
        np.array([10.8, 34.6, 10, 1.1, 7.7, 11.9], dtype=f8)
    ], dtype=d2)

    assert diff(a, b, min_ratio=0.6, atol=0.2, record_compare_names=False, record_compare_data=True) == Diff(
        ratio=2./3,
        diffs=[
            Chunk(data_a=a[:1], data_b=b[:1], eq=False),
            Chunk(data_a=a[1:2], data_b=b[1:2], eq=True, details=[array('i', [1, 1, 1])]),
            Chunk(data_a=a[2:3], data_b=b[2:3], eq=False),
            Chunk(data_a=a[3:], data_b=b[3:], eq=True, details=[array('i', [1, 1, 1])] * 3),
        ],
    )
