from collections.abc import Sequence, MutableSequence, Generator
from typing import Optional, Union
from array import array
from itertools import groupby

from .chunk import Diff, Chunk
from .myers import search_graph_recursive as pymyers, MAX_COST, MAX_CALLS, MIN_RATIO
from .cython.cmyers import search_graph_recursive as cmyers
from sdiff.protocols import wrap

_nested_containers = (list, tuple)

try:
    import numpy
except ImportError:
    numpy = None
else:
    _nested_containers = (*_nested_containers, numpy.ndarray)


_kernels = {
    None: cmyers,
    "c": cmyers,
    "py": pymyers,
}


def diff(
        a: Sequence[object],
        b: Sequence[object],
        eq=None,
        atol: Optional[float] = None,
        struct_threshold: Optional[int] = None,
        struct_mask: Optional[Sequence[bool]] = None,
        min_ratio: float = MIN_RATIO,
        max_cost: int = MAX_COST,
        max_calls: int = MAX_CALLS,
        eq_only: bool = False,
        kernel: Optional[str] = None,
        rtn_diff: Union[bool, array] = True,
        dig=None,
        strict: bool = True,
        no_python: bool = False,
) -> Diff:
    """
    Computes a diff between sequences.

    Parameters
    ----------
    a
        The first sequence.
    b
        The second sequence.
    eq
        Equality measure. Can be either of these:
        - a function ``fun(i, j) -> float`` telling the similarity ratio
          from 0 (dissimilar) to 1 (same).
        - a pair of sequences ``(a_, b_)`` substituting the input sequences
          when computing the diff. The returned chunks, however, are still
          composed of elements from a and b.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    struct_threshold
        For arrays of struct data types (e.g. numpy record arrays) specifies
        the minimal number of fields to be equal for the whole struct to be
        considered equal.
    struct_mask
        For arrays of struct data types (e.g. numpy record arrays) specifies
        a mask including/removing individual fields into/from comparison.
        Masks for inner (nested) struct data types are not supported.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
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
        script. This provides an early stop and further savings in run times
        is some cases. If set, enforces rtn_diff=False.
    kernel
        The kernel to use:
        - 'py': python implementation of Myers diff algorithm
        - 'c': cython implementation of Myers diff algorithm
    rtn_diff
        If True, computes and returns the diff. Otherwise, returns the
        similarity ratio only. Computing the similarity ratio only is
        typically faster and consumes less memory.
        This option also accepts an array: if an array passed will
        perform a full diff and store codes into the provided array
        while the returned object will be the same as rtn_diff=False.
    dig
        An optional callable(i, j) comparing two elements in details.
        This callable will be used to populate the "details" field
        of the resulting diff.
    strict
        If True, ensures that the returned diff either satisfies both
        min_ratio and max_cost or otherwise has a zero ratio.
    no_python
        If True will disallow slow python-based comparison protocols.

    Returns
    -------
    A diff object describing the diff.
    """
    if eq_only:
        rtn_diff = False
    n = len(a)
    m = len(b)
    if eq is None:
        eq = (a, b)
    if isinstance(eq, tuple):
        _a, _b = eq
        assert len(_a) == n
        assert len(_b) == m
    if isinstance(rtn_diff, array):
        codes = rtn_diff
        rtn_diff = False
    elif rtn_diff:
        codes = array('b', b'\xFF' * (n + m))
    else:
        codes = None

    _kernel = _kernels[kernel]
    backend = wrap(
        arg=eq,
        allow_python=not no_python,
        atol=atol,
        resolver=dig,
        struct_threshold=struct_threshold,
        struct_mask=struct_mask,
    )

    total_len = n + m
    if total_len == 0:
        return Diff(ratio=1, diffs=[])

    max_cost = min(max_cost, int(total_len - total_len * min_ratio))

    cost = _kernel(
        n=n,
        m=m,
        comparison_backend=backend,
        max_cost=max_cost,
        eq_only=eq_only,
        max_calls=max_calls,
        out=codes,
    )

    if strict and cost > max_cost:
        if rtn_diff:
            return Diff(ratio=0, diffs=[Chunk(data_a=a, data_b=b, eq=False)])
        else:
            return Diff(ratio=0, diffs=None)

    ratio = (total_len - cost) / total_len
    if rtn_diff:
        canonize(codes)
        return Diff(
            ratio=ratio,
            diffs=list(codes_to_chunks(a, b, codes, dig=backend.resolve)),
        )
    else:
        return Diff(ratio=ratio, diffs=None)


def canonize(codes: MutableSequence[int]):
    """
    Canonize the codes sequence in-place.

    Parameters
    ----------
    codes
        A sequence of diff codes.
    """
    n_horizontal = n_vertical = 0
    n = len(codes)
    for code_i in range(n + 1):
        if code_i != n:
            code = codes[code_i] % 4
        else:
            code = 0
        if code == 1:
            n_horizontal += 1
        elif code == 2:
            n_vertical += 1
        elif n_horizontal + n_vertical:
            for i in range(code_i - n_horizontal - n_vertical, code_i - n_vertical):
                codes[i] = 1
            for i in range(code_i - n_vertical, code_i):
                codes[i] = 2
            n_horizontal = n_vertical = 0


def codes_to_chunks(a: Sequence, b: Sequence, codes: Sequence[int], dig=None) -> Generator[Chunk, None, None]:
    """
    Given the original sequences and diff codes, produces diff chunks.

    Parameters
    ----------
    a
    b
        The original sequences.
    codes
        Diff codes.
    dig
        A function to re-compute per-element diff for equal chunks.

    Returns
    -------
    A list of diff chunks.
    """
    offset_a = offset_b = 0
    for neq, code_group in groupby((
        code
        for code in codes
        if code != 0),
        key=lambda x: bool(x % 3),
    ):
        eq = not neq
        n = offset_a
        m = offset_b
        for code in code_group:
            n += code % 2
            m += code // 2

        _a = a[offset_a:n]
        _b = b[offset_b:m]
        details = None
        if eq and dig is not None:
            try:
                details = [
                    dig(i, j)
                    for i, j in zip(range(offset_a, n), range(offset_b, m))
                ]
            except NotImplementedError:
                dig = None

        yield Chunk(
            data_a=_a,
            data_b=_b,
            eq=eq,
            details=details,
        )

        offset_a = n
        offset_b = m


def _pop_optional(seq):
    if isinstance(seq, Sequence):
        return seq[0], seq[1:] if len(seq) > 1 else seq
    else:
        return seq, seq


def diff_nested(
        a,
        b,
        eq=None,
        atol: Optional[float] = None,
        struct_threshold: Optional[int] = None,
        struct_mask: Optional[Sequence[bool]] = None,
        min_ratio: Union[float, tuple[float, ...]] = MIN_RATIO,
        max_cost: Union[int, tuple[int, ...]] = MAX_COST,
        max_calls: Union[int, tuple[int, ...]] = MAX_CALLS,
        eq_only: bool = False,
        kernel: Optional[str] = None,
        rtn_diff: Union[bool, array] = True,
        nested_containers: tuple = _nested_containers,
        max_depth: int = 0xFF,
        _blacklist_a: set = frozenset(),
        _blacklist_b: set = frozenset(),
) -> Diff:
    """
    Computes a diff between nested sequences.

    Parameters
    ----------
    a
        The first nested sequence.
    b
        The second nested sequence.
    eq
        An optional pair of sequences ``(a_, b_)`` substituting the input
        sequences when computing the diff. The returned chunks, however, are
        still composed of elements from a and b.
    atol
        If set, will use an approximate condition ``abs(a[i] - b[j]) <= atol``
        instead of the equality comparison ``a[i] == b[j]``.
    struct_threshold
        For arrays of struct data types (e.g. numpy record arrays) specifies
        the minimal number of fields to be equal for the whole struct to be
        considered equal.
    struct_mask
        For arrays of struct data types (e.g. numpy record arrays) specifies
        a mask including/removing individual fields into/from comparison.
        Masks for inner (nested) struct data types are not supported.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which sub-sequences are considered "equal".
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
        This option also accepts an array: if an array passed will
        perform a full diff and store codes into the provided array
        while the returned object will be the same as rtn_diff=False.
    nested_containers
        A collection of types that are considered to be capable of nesting.
    max_depth
        Maximal recursion depth while exploring a and b.
    _blacklist_a
    _blacklist_b
        Collections with object ids tracking possible circular references.

    Returns
    -------
    A diff object describing the diff.
    """
    if eq_only:
        rtn_diff = False
    a_ = a
    b_ = b
    if eq is not None:
        a_, b_ = eq

    min_ratio_here, min_ratio_pass = _pop_optional(min_ratio)
    max_cost_here, max_cost_pass = _pop_optional(max_cost)
    max_calls_here, max_calls_pass = _pop_optional(max_calls)

    if max_depth <= 1:
        return diff(
            a,
            b,
            eq=eq,
            atol=atol,
            struct_threshold=struct_threshold,
            struct_mask=struct_mask,
            min_ratio=min_ratio_here,
            max_cost=max_cost_here,
            max_calls=max_calls_here,
            eq_only=eq_only,
            kernel=kernel,
            rtn_diff=rtn_diff,
        )

    if (container_type := type(a_)) is type(b_):
        if container_type in nested_containers:

            if id(a_) in _blacklist_a or id(b_) in _blacklist_b:
                raise ValueError("encountered recursive nesting of inputs")
            _blacklist_a = {*_blacklist_a, id(a_)}
            _blacklist_b = {*_blacklist_b, id(b_)}

            def _eq(i: int, j: int):
                return diff_nested(
                    a=a[i],
                    b=b[j],
                    eq=(a_[i], b_[j]),
                    atol=atol,
                    struct_threshold=struct_threshold,
                    struct_mask=struct_mask,
                    min_ratio=min_ratio_pass,
                    max_cost=max_cost_pass,
                    max_calls=max_calls_pass,
                    eq_only=True,
                    kernel=kernel,
                    nested_containers=nested_containers,
                    max_depth=max_depth - 1,
                    _blacklist_a=_blacklist_a,
                    _blacklist_b=_blacklist_b,
                )

            if rtn_diff and not isinstance(rtn_diff, array):
                def _dig(i: int, j: int):
                    return diff_nested(
                        a=a[i],
                        b=b[j],
                        eq=(a_[i], b_[j]),
                        atol=atol,
                        struct_threshold=struct_threshold,
                        struct_mask=struct_mask,
                        min_ratio=min_ratio_pass,
                        max_cost=max_cost_pass,
                        max_calls=max_calls_pass,
                        eq_only=False,
                        kernel=kernel,
                        rtn_diff=rtn_diff,
                        nested_containers=nested_containers,
                        max_depth=max_depth - 1,
                        _blacklist_a=_blacklist_a,
                        _blacklist_b=_blacklist_b,
                    )
            else:
                _dig = None

        elif issubclass(container_type, Sequence):  # inputs are containers but we do not recognize them as, potentially, nested
            _eq = (a_, b_)
            _dig = None

        else:  # inputs are not containers
            return bool(a_ == b_)

    else:  # inputs are not the same type
        return bool(a_ == b_)

    result = diff(
        a=a,
        b=b,
        eq=_eq,
        atol=atol,
        struct_threshold=struct_threshold,
        struct_mask=struct_mask,
        min_ratio=min_ratio_here,
        max_cost=max_cost_here,
        max_calls=max_calls_here,
        eq_only=eq_only,
        kernel=kernel,
        rtn_diff=rtn_diff,
        dig=_dig,
        strict=True,
    )

    return result
