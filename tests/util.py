from sdiff.chunk import Chunk
from sdiff.numpy import NumpyDiff


def np_chunk_eq(a: Chunk, b: Chunk) -> bool:
    return a.data_a.shape == b.data_a.shape and a.data_b.shape == b.data_b.shape and (a.data_a == b.data_a).all() and (a.data_b == b.data_b).all() and a.eq == b.eq


def np_raw_diff_eq(a: NumpyDiff, b:NumpyDiff) -> bool:
    return a.a.shape == b.a.shape and a.b.shape == b.b.shape and a.eq.shape == b.eq.shape and (a.a == b.a).all() and (a.b == b.b).all() and (a.eq == b.eq).all() and a.row_diff_sig == b.row_diff_sig and a.col_diff_sig == b.col_diff_sig
