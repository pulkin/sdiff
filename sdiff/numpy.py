from typing import Optional, Union, NamedTuple
from collections.abc import Sequence
from itertools import groupby
from functools import partial

import numpy as np

from .chunk import Diff, Chunk, ChunkSignature, Signature
from .myers import MAX_COST, MAX_CALLS, MIN_RATIO
from .sequence import diff_nested, diff as sequence_diff, _pop_optional
from .cython.compare import ComparisonBackend
from .protocols import wrap


def diff(
        a,
        b,
        eq=None,
        atol: Optional[float] = None,
        min_ratio: Union[float, tuple[float]] = MIN_RATIO,
        max_cost: Union[int, tuple[int]] = MAX_COST,
        max_calls: Union[int, tuple[int]] = MAX_CALLS,
        eq_only: bool = False,
        kernel: Optional[str] = None,
        rtn_diff: bool = True,
) -> Diff:
    """
    Computes a diff between numpy tensors of equal rank.

    Parameters
    ----------
    a
        The first tensor.
    b
        The second tensor.
    eq
        An optional pair of tensors ``(a_, b_)`` substituting the input
        tensors when computing the diff. The returned chunks, however, are
        still composed of elements from a and b.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which arrays are considered "equal".
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned elements in both sequences. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_calls
        The maximal number of calls (iterations) after which the algorithm gives
        up. This has to be lower than ``len(a) * len(b)`` to have any effect.
    eq_only
        If True, attempts to guarantee the existence of an edit script
        satisfying both min_ratio and max_cost without actually finding the
        script. This provides an early stop is some cases and further savings
        on run times. Will enforce rtn_diff=False.
    kernel
        The kernel to use:
        - 'py': python implementation of Myers diff algorithm
        - 'c': cython implementation of Myers diff algorithm
    rtn_diff
        If True, computes and returns the diff. Otherwise, returns the
        similarity ratio only. Computing the similarity ratio only is
        typically faster and consumes less memory.

    Returns
    -------
    A diff object describing the diff.
    """
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.ndim != b.ndim:
        raise ValueError(f"{a.ndim=} != {b.ndim=}")

    ndim = a.ndim

    return diff_nested(
        a=a,
        b=b,
        eq=eq,
        atol=atol,
        min_ratio=min_ratio,
        max_cost=max_cost,
        max_calls=max_calls,
        eq_only=eq_only,
        kernel=kernel,
        rtn_diff=rtn_diff,
        nested_containers=(np.ndarray,),
        max_depth=ndim,
    )


def _to_record(arg):
    if isinstance(arg, np.ndarray):
        if arg.dtype.names is None:
            return np.rec.fromarrays([arg], dtype=np.dtype([("f", arg.dtype)]))
    elif isinstance(arg, np.dtype):
        if arg.names is None:
            return np.dtype([("f", arg)])
    else:
        raise ValueError(f"unbknown argument: {arg}")
    return arg


def dtype_diff(
        a,
        b,
        names: bool = False,
        min_ratio: Union[float, tuple[float]] = MIN_RATIO,
        max_cost: Union[int, tuple[int]] = MAX_COST,
        max_calls: Union[int, tuple[int]] = MAX_CALLS,
        eq_only: bool = False,
        kernel: Optional[str] = None,
        rtn_diff: bool = True,
        data: bool = True,
        data_atol: Optional[float] = None,
        data_min_ratio: float = MIN_RATIO,
        data_max_cost: int = MAX_COST,
) -> Diff:
    """
    Computes a diff between numpy record dtypes.
    Produces a diff between record fields of the two dtypes.

    Parameters
    ----------
    a
    b
        Numpy dtypes.
    names
        If True, requires field names to be the same.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which arrays are considered "equal".
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned elements in both sequences. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_calls
        The maximal number of calls (iterations) after which the algorithm gives
        up. This has to be lower than ``len(a) * len(b)`` to have any effect.
    eq_only
        If True, attempts to guarantee the existence of an edit script
        satisfying both min_ratio and max_cost without actually finding the
        script. This provides an early stop is some cases and further savings
        on run times. Will enforce rtn_diff=False.
    kernel
        The kernel to use:
        - 'py': python implementation of Myers diff algorithm
        - 'c': cython implementation of Myers diff algorithm
    rtn_diff
        If True, computes and returns the diff. Otherwise, returns the
        similarity ratio only. Computing the similarity ratio only is
        typically faster and consumes less memory.
    data
        If True, and both a, b are arrays, will use their data to align fields.
    data_atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]`` for comparing data.
    data_min_ratio
        min_ratio for the data.
    data_max_cost
        max_cost for the data.

    Returns
    -------
    The resulting diff between records' fields.
    """
    a = _to_record(a)
    b = _to_record(b)

    data_a = data_b = None

    if isinstance(a, np.ndarray):
        data_a, a = a, a.dtype
    else:
        data = False

    if isinstance(b, np.ndarray):
        data_b, b = b, b.dtype
    else:
        data = False

    if (data_a is None) != (data_b is None):
        raise ValueError("a and b have to be both arrays or both dtypes")

    fields_a = list((name, t[0]) for name, t in a.fields.items())
    fields_b = list((name, t[0]) for name, t in b.fields.items())

    if data:
        # a comparison taking into account field data
        if names:
            def _eq(i: int, j: int) -> bool:
                name_a, type_a = fields_a[i]
                name_b, type_b = fields_b[j]
                return (
                    type_a == type_b and
                    name_a == name_b and
                    sequence_diff(
                        a=data_a[name_a],
                        b=data_b[name_b],
                        rtn_diff=False,
                        eq_only=True,
                        atol=data_atol,
                        min_ratio=data_min_ratio,
                        max_cost=data_max_cost,
                    )
                )
        else:
            def _eq(i: int, j: int) -> bool:
                name_a, type_a = fields_a[i]
                name_b, type_b = fields_b[j]
                return (
                    type_a == type_b and
                    sequence_diff(
                        a=data_a[name_a],
                        b=data_b[name_b],
                        rtn_diff=False,
                        eq_only=True,
                        atol=data_atol,
                        min_ratio=data_min_ratio,
                        max_cost=data_max_cost,
                    )
                )
    else:
        # a simple comparison not involving data
        _eq = None
        if not names:
            _eq = ([v for _, v in fields_a], [v for _, v in fields_b])
    return sequence_diff(
        fields_a,
        fields_b,
        eq=_eq,
        min_ratio=min_ratio,
        max_cost=max_cost,
        max_calls=max_calls,
        eq_only=eq_only,
        kernel=kernel,
        rtn_diff=rtn_diff,
    )


def align_inflate_arrays(a: np.ndarray, b: np.ndarray, field_diff: Diff) -> tuple[np.ndarray, np.ndarray]:
    """
    Aligns arrays towards a common (record) dtype.
    Given two record arrays and the dtype diff, produces the corresponding pair of arrays with the same record
    dtype containing fields from both arrays.

    Parameters
    ----------
    a
    b
        The arrays to align.
    field_diff
        A diff telling how to align fields.

    Returns
    -------
    A pair of aligned inflated arrays.
    """
    a = _to_record(a)
    b = _to_record(b)

    dtype_a, dtype_b = map(np.dtype, map(list, field_diff.get_inflated_ab()))
    init_a, init_b = field_diff.with_data(
        data_a=[a[field] for field in a.dtype.fields],
        data_b=[b[field] for field in b.dtype.fields],
    ).get_inflated_ab(
        hook_a_b=partial(np.zeros_like, shape=a.shape),
        hook_b_a=partial(np.zeros_like, shape=b.shape),
    )

    return np.rec.fromarrays(init_a, dtype=dtype_a), np.rec.fromarrays(init_b, dtype=dtype_b)


def common_diff_sig(n: int, m: int, diffs: Sequence[Diff]) -> Signature:
    """
    Computes a "common" diff signature using breadth-first search.

    Parameters
    ----------
    n
    m
        Dimensions of arrays.
    diffs
        A sequence of diffs.

    Returns
    -------
    The signature as a tuple of tuples with chunk lengths and their
    equality statuses (equal/not equal).
    """
    if n == 0 and m == 0:
        return Signature(parts=tuple())
    if n == 0 or m == 0:
        return Signature(parts=(ChunkSignature(size_a=n, size_b=m, eq=False),))

    # prepare search space
    space = np.zeros((n, m), dtype=int)
    for diff in diffs:
        x = y = 0
        for chunk in diff.diffs:
            if chunk.eq:
                for i in range(len(chunk.data_a)):
                    space[x + i, y + i] += 1
            x += len(chunk.data_a)
            y += len(chunk.data_b)

    # search
    for y in range(m):
        if y == 0:
            for x in range(1, n):
                space[x, y] = max(space[x, y], space[x - 1, y])
        elif n > 0:
            space[0, y] = max(space[0, y], space[0, y - 1])
            for x in range(1, n):
                space[x, y] = max(space[x - 1, y], space[x, y - 1], space[x - 1, y - 1] + space[x, y])

    # trace back
    x = n - 1
    y = m - 1
    is_b = np.zeros(n + m, dtype=bool)
    is_eq = np.zeros(n + m + 2, dtype=bool)
    pos = n + m
    while x >= 0 and y >= 0:
        if x > 0 and space[x, y] == space[x - 1, y]:
            x -= 1
            pos -= 1
        elif y > 0 and space[x, y] == space[x, y - 1]:
            y -= 1
            pos -= 1
            is_b[pos] = 1
        else:
            is_eq[pos] = is_eq[pos - 1] = 1
            x -= 1
            y -= 1
            pos -= 2
            is_b[pos + 1] = 1
    x += 1
    y += 1
    is_b[x:x + y] = 1
    is_eq[0] = 1 - is_eq[1]
    is_eq[-1] = 1 - is_eq[-2]
    ix, = np.nonzero(is_eq[1:] != is_eq[:-1])
    return Signature(parts=tuple(
        ChunkSignature(
            size_a=int((~is_b[fr:to]).sum()),
            size_b=int((is_b[fr:to]).sum()),
            eq=bool(is_eq[fr + 1]),
        )
        for fr, to in zip(ix[:-1], ix[1:])
    ))


def get_row_col_diff(
        a: np.ndarray,
        b: np.ndarray,
        atol: Optional[float] = None,
        min_ratio: Union[float, tuple[float]] = MIN_RATIO,
        max_cost: Union[int, tuple[int]] = MAX_COST,
        max_calls: Union[int, tuple[int]] = MAX_CALLS,
        kernel: Optional[str] = None,
) -> tuple[Signature, Signature]:
    """
    Aligns rows and columns of two matrices and returns the corresponding pair
    of diffs.

    The algorithm first computes a generic diff between the two matrices to
    determine the aligned rows. It will then use the diff withing the aligned
    rows to determine the "common" or "average" diff between the columns.

    Parameters
    ----------
    a
        The first matrix.
    b
        The second matrix.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which arrays are considered "equal".
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned elements in both sequences. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_calls
        The maximal number of calls (iterations) after which the algorithm gives
        up. This has to be lower than ``len(a) * len(b)`` to have any effect.
    kernel
        The kernel to use:
        - 'py': python implementation of Myers diff algorithm
        - 'c': cython implementation of Myers diff algorithm

    Returns
    -------
    Two sequences describing row and column diffs.
    """
    if a.ndim != 2:
        raise ValueError(f"{a.ndim=} is not a matrix")
    if b.ndim != 2:
        raise ValueError(f"{b.ndim=} is not a matrix")

    base_diff = diff(
        a=a,
        b=b,
        atol=atol,
        min_ratio=min_ratio,
        max_cost=max_cost,
        max_calls=max_calls,
        kernel=kernel,
    )

    row_sig = base_diff.signature
    in_row_diff = []
    for chunk in base_diff.diffs:
        if chunk.eq:
            in_row_diff.extend(chunk.details)
    col_sig = common_diff_sig(a.shape[1], b.shape[1], in_row_diff)
    return row_sig, col_sig


def align_inflate(a: np.ndarray, b: np.ndarray, val, sig: Signature, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Align arrays with the given value by inflating them.

    Parameters
    ----------
    a
    b
        The arrays to align.
    val
        The scalar value to inflate with.
    sig
        The diff signature to use.
    dim
        The dimension to inflate.

    Returns
    -------
    The inflated array.
    """
    assert (ndim := a.ndim) == b.ndim

    s = len(sig)
    a_shape = list(a.shape)
    b_shape = list(b.shape)
    a_shape[dim] = s
    b_shape[dim] = s
    result_a = np.full(a_shape, val, dtype=a.dtype)
    result_b = np.full(b_shape, val, dtype=b.dtype)

    pre = (slice(None),) * dim
    post = (slice(None),) * (ndim - dim - 1)

    offset_a = offset_b = offset = 0
    for chunk in sig.parts:
        # a comes first
        result_a[(*pre, slice(offset, offset + chunk.size_a), *post)] = (
            a[(*pre, slice(offset_a, offset_a + chunk.size_a), *post)]
        )
        offset_a += chunk.size_a
        if not chunk.eq:
            offset += chunk.size_a
        # b is second
        result_b[(*pre, slice(offset, offset + chunk.size_b), *post)] = (
            b[(*pre, slice(offset_b, offset_b + chunk.size_b), *post)]
        )
        offset_b += chunk.size_b
        offset += chunk.size_b
    return result_a, result_b


class NumpyDiff(NamedTuple):
    """
    Three 2D arrays of the same shape describing an aligned diff
    between two arrays.

    Parameters
    ----------
    a
    b
        Inflated arrays.
    eq
        Per-element equivalence between the arrays.
    row_diff_sig
    col_diff_sig
        Row and column diff signatures.
    """
    a: np.ndarray
    b: np.ndarray
    eq: np.ndarray
    row_diff_sig: Signature
    col_diff_sig: Signature

    @property
    def ratio(self) -> float:
        """Diff ratio: the fraction of aligned rows in the total row count"""
        eq = total = 0
        for sig in self.row_diff_sig.parts:
            n = sig.size_a + sig.size_b
            total += n
            eq += n * sig.eq
        if eq == 0 and total == 0:
            return 1.
        return eq / total

    @property
    def aligned_ratio(self) -> float:
        """Aligned ratio: the fraction of equal elements in aligned matrices"""
        return self.eq.mean(dtype=float)

    @property
    def a_shape(self) -> tuple[int, int]:
        """The original shape of a"""
        return sum(i.size_a for i in self.row_diff_sig.parts), sum(i.size_a for i in self.col_diff_sig.parts)

    @property
    def b_shape(self) -> tuple[int, int]:
        """The original shape of b"""
        return sum(i.size_b for i in self.row_diff_sig.parts), sum(i.size_b for i in self.col_diff_sig.parts)

    def to_plain(self) -> Diff:
        """
        Composes the diff.

        Returns
        -------
        The diff.
        """
        rows_a, rows_b = align_inflate(  # two masks telling if it is a true row or an inflated one
            a=np.ones(len(self.a), dtype=bool),
            b=np.ones(len(self.b), dtype=bool),
            val=0,
            sig=self.row_diff_sig,
            dim=0,
        )
        aligned_mask = rows_a * rows_b
        equal_mask = self.eq.all(axis=1)

        chunks = []
        offset = 0
        for key, group in groupby(aligned_mask.astype(int) + equal_mask):
            group_size = sum(1 for _ in group)
            to = offset + group_size

            if key == 0:  # not equal
                chunks.append(Chunk(
                    data_a=self.a[offset:to][rows_a[offset:to]],
                    data_b=self.b[offset:to][rows_b[offset:to]],
                    eq=False,
                ))

            else:  # aligned or plain equal
                chunks.append(Chunk(
                    data_a=self.a[offset:to][rows_a[offset:to]],
                    data_b=self.b[offset:to][rows_b[offset:to]],
                    eq=True,
                    details=None if key == 2 else self.eq[offset:to],
                ))
            offset = to

        return Diff(
            ratio=self.ratio,
            diffs=chunks,
        )


_undefined = object()


def diff_aligned_2d(
        a: np.ndarray,
        b: np.ndarray,
        fill,
        eq=None,
        atol: Optional[float] = None,
        fill_eq=_undefined,
        min_ratio: Union[float, tuple[float, ...]] = MIN_RATIO,
        max_cost: Union[int, tuple[int, ...]] = MAX_COST,
        max_calls: Union[int, tuple[int, ...]] = MAX_CALLS,
        col_diff_sig: Optional[Signature] = None,
        kernel: Optional[str] = None,
) -> NumpyDiff:
    """
    Computes an aligned diff between numpy matrices.

    Parameters
    ----------
    a
        The first matrix.
    b
        The second matrix.
    fill
        The empty value to use when filling both matrices.
    eq
        An optional pair of tensors ``(a_, b_)`` substituting the input
        matrices when computing the diff. The returned chunks, however, are
        still composed of elements from a and b.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    fill_eq
        The empty value to use when filling "eq" matrices.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which arrays are considered "equal".
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned elements in both sequences. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_calls
        The maximal number of calls (iterations) after which the algorithm gives
        up. This has to be lower than ``len(a) * len(b)`` to have any effect.
    col_diff_sig
        The column diff signature. If provided, this dramatically speeds up
        the diff computation by assuming the column alignment via the provided
        signature. In practice, this is useful for tables where column alignment
        can be deduced by comparing column names separately from table contents.
    kernel
        The kernel to use:
        - 'py': python implementation of Myers diff algorithm
        - 'c': cython implementation of Myers diff algorithm

    Returns
    -------
    A named tuple with the following fields:
    - ``result.a``: inflated matrix a
    - ``result.b``: inflated matrix b
    - ``result.eq``: equality matrix
    - ``result.row_diff_signature``: a signature describing the alignment of rows in a and b;
    - ``result.col_diff_signature``: a signature describing the alignment of cols in a and b;

    Unlike other diff routines, numpy diff returns an intermediate representation of diff.
    You can convert it to the common ``Diff`` type through ``result.to_diff``.
    """
    a_, b_ = a, b
    if eq is not None:
        a_, b_ = eq
        if fill_eq is _undefined:
            fill_eq = fill
    if col_diff_sig:
        # column diff provided: run a faster algorithm using column-aligned data
        min_ratio_here, min_ratio = _pop_optional(min_ratio)
        min_ratio_row, _ = _pop_optional(min_ratio)

        max_cost_here, max_cost = _pop_optional(max_cost)
        max_cost_row, _ = _pop_optional(max_cost)
        max_calls_here, _ = _pop_optional(max_calls)

        # align columns  using the provided column diff
        a, b = align_inflate(a, b, fill, col_diff_sig, 1)
        if eq is not None:
            a_, b_ = align_inflate(a_, b_, fill_eq, col_diff_sig, 1)
        else:
            a_, b_ = a, b
        # compute a mask telling which columns can be compared
        # and which ones have to be ignored
        mask = np.empty(len(col_diff_sig), dtype=float)
        offset = 0
        for chunk in col_diff_sig.parts:
            delta = len(chunk)
            mask[offset:offset + delta] = chunk.eq + 1
            offset += delta

        _max_cost = mask.sum()
        min_ratio_row = max(min_ratio_row, (_max_cost - max_cost_row) / _max_cost)
        mask *= len(mask) / _max_cost

        # crunch row differences without using the shallow algorithm
        raw_diff = sequence_diff(
            a=a_,
            b=b_,
            eq=wrap(
                (np.core.records.fromarrays(a_.T), np.core.records.fromarrays(b_.T)),
                atol=atol,
                struct_weights=mask,
                struct_threshold=min_ratio_row,
            ),
            min_ratio=min_ratio_here,
            max_cost=max_cost_here,
            max_calls=max_calls_here,
            kernel=kernel,
            no_python=True,
        )
        row_diff_sig = raw_diff.signature

        # finally, align rows
        a, b = align_inflate(a, b, fill, row_diff_sig, 0)
        if eq is not None:
            a_, b_ = align_inflate(a_, b_, fill_eq, row_diff_sig, 0)
        else:
            a_, b_ = a, b
        signatures = row_diff_sig, col_diff_sig

    else:
        # generic row-column diff
        signatures = get_row_col_diff(
            a=a_,
            b=b_,
            atol=atol,
            min_ratio=min_ratio,
            max_cost=max_cost,
            max_calls=max_calls,
            kernel=kernel,
        )
        for dim, sig in enumerate(signatures):
            a, b = align_inflate(a, b, fill, sig, dim)
            if eq is not None:
                a_, b_ = align_inflate(a_, b_, fill_eq, sig, dim)
        if eq is None:
            a_, b_ = a, b

    # a, b, a_, b_ are all of the same exact shape starting here
    if atol is None:
        eq_matrix = a_ == b_
    else:
        eq_matrix = np.abs(a_ - b_) <= atol
    idx = tuple()
    for dim, sig in enumerate(signatures):
        offset = 0
        for part in sig.parts:
            if not part.eq:
                eq_matrix[(*idx, slice(offset, offset + (n := part.size_a + part.size_b)))] = False
                offset += n
            else:
                offset += part.size_a
        idx = (*idx, slice(None))
    return NumpyDiff(
        a=a,
        b=b,
        eq=eq_matrix,
        row_diff_sig=signatures[0],
        col_diff_sig=signatures[1],
    )
