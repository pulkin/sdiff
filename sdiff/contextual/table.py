from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
from numpy.random.mtrand import Sequence

from ..chunk import Signature
from ..numpy import NumpyDiff, align_inflate, diff_aligned_2d
from ..sequence import MAX_COST, MIN_RATIO
from ..sequence import diff as diff_sequence
from .base import AnyDiff, profile

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class Columns:
    a: Sequence[str]
    b: Sequence[str]
    """
    Column names in the table diff.

    Parameters
    ----------
    a
    b
        Column names in a and b.
    """

    def is_eq(self) -> bool:
        return all(i == j for i, j in zip(self.a, self.b, strict=False))


@dataclass
class TableDiff(AnyDiff):
    data: NumpyDiff
    columns: Columns | None = None
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A diff between tables.

    Parameters
    ----------
    name
        A name this diff belongs to.
    data
        Diff data.
    columns
        Optional column names.
    stats
        Stats associated with this diff.
    """

    def is_eq(self) -> bool:
        return bool(self.data.eq.all()) and (
            self.columns is None or self.columns.is_eq()
        )


@profile("2D comparison")
def diff(
    a,
    b,
    name: str,
    min_ratio: float = MIN_RATIO,
    min_ratio_row: float = MIN_RATIO,
    max_cost: int = MAX_COST,
    max_cost_row: int = MAX_COST,
    fill="",
    fill_col_names="",
    columns: Signature | str | tuple[list[str], list[str]] | None = "columns",
    dtype: type = object,
) -> TableDiff:
    """
    Computes a diff between tables.

    Parameters
    ----------
    a
    b
        Tables to compare.
    name
        A name for the resulting diffs.
    min_ratio
    min_ratio_row
    max_cost
    max_cost_row
        Constants specifying the maximal allowed complexity
        of the produced diffs, see the description in
        ``sdiff.sequence.diff``.
    fill
        An object to fill aligned tables with.
    fill_col_names
        An object to fill aligned column names with.
    columns
        If specified, will use the provided diff signature to
        align columns. If a pair of lists (column names) is provided,
        will compute the signature by comparing the provided lists.
        In case of the provided string, column names will be taken
        from the corresponding attribute in a and b. This will fall
        back to the (slow) recursive comparison if the attribute
        does not exist.
    dtype
        Which type to use when casting inputs into dense numpy arrays.

    Returns
    -------
    The table diff.
    """
    if isinstance(columns, str):
        try:
            columns = (getattr(a, columns), getattr(b, columns))
        except AttributeError:
            columns = None

    cols_a = cols_b = None
    if isinstance(columns, tuple):
        cols_a, cols_b = columns
        columns = diff_sequence(cols_a, cols_b, min_ratio=0).signature

    if not (t_a := type(a)) == (t_b := type(b)):
        raise ValueError(f"type(a)={t_a} != type(b)={t_b}")

    def _hash(_a: np.ndarray) -> np.ndarray:
        # TODO might need a faster hash
        return np.array(list(map(hash, _a.flatten())), dtype=int).reshape(_a.shape)

    if issubclass(t_a, np.ndarray):
        a = a.astype(dtype)
        b = b.astype(dtype)

    elif pd is not None and issubclass(t_a, pd.DataFrame):
        a = a.to_numpy(dtype=dtype)
        b = b.to_numpy(dtype=dtype)

    else:
        raise ValueError(f"unknown input type: {t_a}")

    eq = tuple(map(_hash, (a, b)))

    np_diff = diff_aligned_2d(
        a=a,
        b=b,
        eq=eq,
        fill=fill,
        fill_eq=0,
        min_ratio=(min_ratio, min_ratio_row),
        max_cost=(max_cost, max_cost_row),
        col_diff_sig=columns,
    )

    return TableDiff(
        name=name,
        data=np_diff,
        columns=Columns(
            *align_inflate(
                a=np.array(cols_a, dtype=str),
                b=np.array(cols_b, dtype=str),
                val=fill_col_names,
                sig=columns,
                dim=0,
            )
        )
        if cols_a is not None
        else None,
    )
