import numpy as np
import pytest

from sdiff.chunk import Diff, Chunk, Signature, ChunkSignature
from sdiff.numpy import diff, get_row_col_diff, align_inflate, diff_aligned_2d, NumpyDiff

from .util import np_chunk_eq, np_chunk_eq_aligned, np_raw_diff_eq


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
            Chunk(data_a=a, data_b=a, eq=True)
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
            Chunk(data_a=a[:1], data_b=b[:1], eq=True),
            Chunk(data_a=a[1:3], data_b=b[1:3], eq=[
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
            Chunk(data_a=a[4:], data_b=b[4:], eq=[
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
    monkeypatch.setattr(Chunk, "__eq__", np_chunk_eq_aligned)

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
            Chunk(data_a=a[2:3], data_b=a1[2:3], eq=eq[2:3]),
            Chunk(data_a=a[3:5], data_b=a1[5:6], eq=False),
            Chunk(data_a=a[6:8], data_b=a1[6:8], eq=eq[6:8]),
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
