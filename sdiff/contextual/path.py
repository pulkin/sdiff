import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, TypeVar, Optional, Sequence, NamedTuple
from dataclasses import dataclass, field
import filecmp
from functools import partial
from collections.abc import Mapping
from numbers import Number
from warnings import warn

import pandas as pd
import numpy as np

from .base import AnyDiff, profile, add_stats
from .text import TextDiff, diff as _diff_text
from .table import TableDiff, diff as _diff_table, Columns
from ..numpy import align_inflate
from ..sequence import MAX_COST, MIN_RATIO

try:
    import magic
    magic_guess_custom = magic.Magic(mime=True, magic_file=str(Path(__file__).parent / "magic"))
except ImportError:
    magic = magic_guess_custom = None

try:
    import pandas
except ImportError:
    pandas = None


DiffKernel = Callable[[Path, Path, str], AnyDiff]
T = TypeVar("T", bound=DiffKernel)


mime_dispatch: dict[str, DiffKernel] = {}


@dataclass
class PathDiff(AnyDiff):
    eq: bool
    message: Optional[str] = None
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A diff indicating that two paths are exact same or different.

    Parameters
    ----------
    name
        A name this diff belongs to.
    eq
        True if paths are the same and False otherwise.
    message
        Whatever message to share.
    stats
        Stats associated with this diff.
    """

    def is_eq(self) -> bool:
        return self.eq


@dataclass
class MIMEDiff(AnyDiff):
    mime_a: str
    mime_b: str
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A diff indicating that two paths have different MIMEs.

    Parameters
    ----------
    name
        A name this diff belongs to.
    mime_a
    mime_b
        The two MIME types.
    stats
        Stats associated with this diff.
    """

    def is_eq(self) -> bool:
        return self.mime_a == self.mime_b


@dataclass
class DeltaDiff(AnyDiff):
    exist_a: bool
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A diff indicating that one of the paths does not exist.

    Parameters
    ----------
    name
        A name this diff belongs to.
    exist_a
        A flag indicating that a path exists in a.
    stats
        Stats associated with this diff.
    """

    @property
    def exist_b(self) -> bool:
        return not self.exist_a

    def is_eq(self) -> bool:
        return False


@dataclass
class CompositeDiff(AnyDiff):
    items: list[AnyDiff]
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A diff with multiple parts.

    Parameters
    ----------
    name
        A name this diff belongs to.
    items
        Diff parts.
    stats
        Stats associated with this diff.
    """

    def is_eq(self) -> bool:
        return all(i.is_eq() for i in self.items)


def mime_kernel(*args: str) -> Callable[[T], T]:
    """
    Associates a diff function with one or more MIME types.

    Parameters
    ----------
    args
        MIME strings.
    """
    def _decorate(kernel: T) -> T:
        for i in args:
            mime_dispatch[i] = kernel
        return kernel
    return _decorate


@mime_kernel("text/plain", "text")
@profile("text fetch")
def diff_text(
        a: Path,
        b: Path,
        name: str,
        min_ratio: float = MIN_RATIO,
        min_ratio_row: float = MIN_RATIO,
        max_cost: int = MAX_COST,
        max_cost_row: int = MAX_COST,
) -> TextDiff:
    """
    Computes a text diff between two files.

    Parameters
    ----------
    a
        The first file path.
    b
        The second file path.
    name
        The name associated with this comparison.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which sub-sequences are considered "equal".
    min_ratio_row
        The ratio above which two lines of text are aligned.
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned lines in both texts. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_cost_row
        The maximal cost below which two lines of text are aligned.

    Returns
    -------
    The text diff.
    """
    with a.open("r") as fa, b.open("r") as fb:
        return _diff_text(list(fa), list(fb), name, min_ratio=min_ratio, min_ratio_row=min_ratio_row,
                          max_cost=max_cost, max_cost_row=max_cost_row)


@profile("pandas preprocessing")
def diff_pd(
        a,
        b,
        name: str,
        min_ratio: float = MIN_RATIO,
        min_ratio_row: float = MIN_RATIO,
        max_cost: int = MAX_COST,
        max_cost_row: int = MAX_COST,
        align_col_data: bool = False,
        table_drop_cols: Optional[Sequence[str]] = None,
        table_sort: Optional[Sequence[str]] = None,
) -> TableDiff:
    """
    Computes a table diff between two ``pandas.DataFrame``.

    Parameters
    ----------
    a
        The first table.
    b
        The second table.
    name
        The name associated with this comparison.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which sub-sequences are considered "equal".
    min_ratio_row
        The ratio above which two lines of text are aligned.
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned lines in both texts. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_cost_row
        The maximal cost below which two lines of text are aligned.
    align_col_data
        For tables, (force) compare columns by data as opposed to their names.
        This may result in much slower comparisons.
    table_drop_cols
        Columns to drop before comparing.
    table_sort
        Sorts tables by the columns specified.

    Returns
    -------
    The table diff.
    """
    if table_drop_cols is not None or table_sort is not None:
        a = a.copy()
        b = b.copy()
        for df in a, b:
            if table_drop_cols is not None:
                df.drop(columns=table_drop_cols, inplace=True, errors="ignore")
            if table_sort is not None:
                df.sort_values(by=table_sort or list(df.columns), inplace=True)
    result = _diff_table(a=a, b=b, name=name, min_ratio=min_ratio, min_ratio_row=min_ratio_row, max_cost=max_cost,
                         max_cost_row=max_cost_row, columns=None if align_col_data else "columns")
    if align_col_data:  # columns were discarded in the diff; add them back
        cols_a, cols_b = align_inflate(np.array(a.columns), np.array(b.columns), "", result.data.col_diff_sig, 0)
        columns = Columns(list(cols_a), list(cols_b))
        result.columns = columns
    return result


if pandas:
    @profile("pandas parsing")
    def diff_pd_simple(
            reader: Callable[[Path], pd.DataFrame],
            a: Path,
            b: Path,
            name: str,
            min_ratio: float = MIN_RATIO,
            min_ratio_row: float = MIN_RATIO,
            max_cost: int = MAX_COST,
            max_cost_row: int = MAX_COST,
            align_col_data: bool = False,
            table_drop_cols: Optional[Sequence[str]] = None,
            table_sort: Optional[Sequence[str]] = None,
    ) -> TableDiff:
        """
        Computes a table diff between two pandas-supported files with tables.

        Parameters
        ----------
        reader
            A reader transforming path into dataframe.
        a
            The first path with a table.
        b
            The second path with a table.
        name
            The name associated with this comparison.
        min_ratio
            The ratio below which the algorithm exits. The values closer to 1
            typically result in faster run times while setting to 0 will force
            the algorithm to crack through even completely dissimilar sequences.
            This affects which sub-sequences are considered "equal".
        min_ratio_row
            The ratio above which two lines of text are aligned.
        max_cost
            The maximal cost of the diff: the number corresponds to the maximal
            count of dissimilar/misaligned lines in both texts. Setting
            this to zero is equivalent to setting min_ratio to 1. The algorithm
            worst-case time complexity scales with this number.
        max_cost_row
            The maximal cost below which two lines of text are aligned.
        align_col_data
            For tables, (force) compare columns by data as opposed to their names.
            This may result in much slower comparisons.
        table_drop_cols
            Columns to drop before comparing.
        table_sort
            Sorts tables by the columns specified.

        Returns
        -------
        The table diff.
        """
        return diff_pd(
            a=reader(a),
            b=reader(b),
            name=name,
            min_ratio=min_ratio,
            min_ratio_row=min_ratio_row,
            max_cost=max_cost,
            max_cost_row=max_cost_row,
            align_col_data=align_col_data,
            table_drop_cols=table_drop_cols,
            table_sort=table_sort,
        )


    diff_pd_csv = mime_kernel("text/csv", "application/csv", "csv")(partial(diff_pd_simple, partial(pd.read_csv, dtype=str, keep_default_na=False, na_filter=False, encoding_errors="replace")))
    diff_pd_feather = mime_kernel("application/vnd.apache.arrow.file")(partial(diff_pd_simple, pd.read_feather))
    diff_pd_parquet = mime_kernel("application/vnd.apache.parquet")(partial(diff_pd_simple, pd.read_parquet))


    @profile("pandas parsing")
    def diff_pd_dict(
            reader: Callable[[Path], dict[str, pd.DataFrame]],
            a: Path,
            b: Path,
            name: str,
            min_ratio: float = MIN_RATIO,
            min_ratio_row: float = MIN_RATIO,
            max_cost: int = MAX_COST,
            max_cost_row: int = MAX_COST,
            align_col_data: bool = False,
            table_drop_cols: Optional[Sequence[str]] = None,
            table_sort: Optional[Sequence[str]] = None,
    ) -> CompositeDiff:
        """
        Computes a table diff between two pandas-supported files with multiple tables.

        Parameters
        ----------
        reader
            A reader transforming path into a dictionary of dataframes.
        a
            The first path with tables.
        b
            The second path with tables.
        name
            The name associated with this comparison.
        min_ratio
            The ratio below which the algorithm exits. The values closer to 1
            typically result in faster run times while setting to 0 will force
            the algorithm to crack through even completely dissimilar sequences.
            This affects which sub-sequences are considered "equal".
        min_ratio_row
            The ratio above which two lines of text are aligned.
        max_cost
            The maximal cost of the diff: the number corresponds to the maximal
            count of dissimilar/misaligned lines in both texts. Setting
            this to zero is equivalent to setting min_ratio to 1. The algorithm
            worst-case time complexity scales with this number.
        max_cost_row
            The maximal cost below which two lines of text are aligned.
        align_col_data
            For tables, (force) compare columns by data as opposed to their names.
            This may result in much slower comparisons.
        table_drop_cols
            Columns to drop before comparing.
        table_sort
            Sorts tables by the columns specified.

        Returns
        -------
        The table diff.
        """
        fmt = "%s/%s"
        a = reader(a)
        b = reader(b)

        for dfs in a, b:
            for df in dfs.values():
                df = df.set_axis(map(str, df.columns), axis=1)
                df.fillna("", inplace=True)

        result = []

        for i in set(a) - set(b):
            result.append(DeltaDiff(fmt % (name, i), True))
        for i in set(b) - set(a):
            result.append(DeltaDiff(fmt % (name, i), False))

        stats = defaultdict(float)
        for i in set(a) & set(b):
            result.append(
                d := diff_pd(
                    a=a[i],
                    b=b[i],
                    name=fmt % (name, i),
                    min_ratio=min_ratio,
                    min_ratio_row=min_ratio_row,
                    max_cost=max_cost,
                    max_cost_row=max_cost_row,
                    align_col_data=align_col_data,
                    table_drop_cols=table_drop_cols,
                    table_sort=table_sort,
                )
            )
            add_stats(d.stats, stats)
        return CompositeDiff(name, result, stats=stats)


    diff_pd_excel = mime_kernel("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel", "excel")(partial(diff_pd_dict, partial(pd.read_excel, dtype=str, keep_default_na=False, na_filter=False, sheet_name=None)))


class GroupedValue(NamedTuple):
    group: Optional[str]
    value: Any


class VariableOption(list):
    def get_value(self, key):
        for pattern, value in self[::-1]:
            if pattern is None or bool(re.fullmatch(pattern, key)):
                return value

    def __repr__(self):
        return f"{type(self).__name__}({super().__repr__()})"

    def __str__(self):
        return f"{type(self).__name__}({super().__str__()})"


@profile("misc")
def diff_path(
        a: Optional[Path],
        b: Optional[Path],
        name: str,
        mime: Optional[str] = None,
        min_ratio: float = MIN_RATIO,
        min_ratio_row: float = MIN_RATIO,
        max_cost: int = MAX_COST,
        max_cost_row: int = MAX_COST,
        align_col_data: bool = False,
        shallow: bool = False,
        table_drop_cols: Optional[Sequence[str]] = None,
        table_sort: Optional[Sequence[str]] = None,
        _replace_dot_name: bool = True,
) -> AnyDiff:
    """
    Computes a diff between two files based on their (common) MIME.

    Parameters
    ----------
    a
        The first file path.
    b
        The second file path.
    name
        The name associated with this comparison.
    mime
        The MIME of the two paths.
    min_ratio
        The ratio below which the algorithm exits. The values closer to 1
        typically result in faster run times while setting to 0 will force
        the algorithm to crack through even completely dissimilar sequences.
        This affects which sub-sequences are considered "equal".
    min_ratio_row
        The ratio above which two lines of text are aligned.
    max_cost
        The maximal cost of the diff: the number corresponds to the maximal
        count of dissimilar/misaligned lines in both texts. Setting
        this to zero is equivalent to setting min_ratio to 1. The algorithm
        worst-case time complexity scales with this number.
    max_cost_row
        The maximal cost below which two lines of text are aligned.
    align_col_data
        For tables, (force) compare columns by data as opposed to their names.
        This may result in much slower comparisons.
    shallow
        If True, performs a shallow comparison without building diffs at all.
    table_drop_cols
        Table columns to drop when comparing tables.
    table_sort
        Sorts tables by the columns specified.
    _replace_dot_name
        Replaces the name="." with a more readable one.

    Returns
    -------
    The diff.
    """
    if _replace_dot_name and name == ".":
        if a is not None and b is not None:
            name = f"{a.name} vs {b.name}"
        elif a is not None:
            name = a.name
        else:
            name = b.name
    def _get_variable_option_val(x):
        if isinstance(x, VariableOption):
            return x.get_value(name)
        return x

    mime = _get_variable_option_val(mime)
    min_ratio = _get_variable_option_val(min_ratio)
    min_ratio_row = _get_variable_option_val(min_ratio_row)
    max_cost = _get_variable_option_val(max_cost)
    max_cost_row = _get_variable_option_val(max_cost_row)
    align_col_data = _get_variable_option_val(align_col_data)
    shallow = _get_variable_option_val(shallow)
    table_drop_cols = _get_variable_option_val(table_drop_cols)
    table_sort = _get_variable_option_val(table_sort)

    if a is None and b is None:
        raise ValueError("either a or b have to be non-None")
    if a is None or b is None:
        return DeltaDiff(name, a is not None)
    if filecmp.cmp(a, b, shallow=False):
        return PathDiff(name, eq=True, message="files are binary equal")
    if shallow:
        return PathDiff(name, eq=False, message="files are not equal (shallow comparison)")
    if mime is None:
        if magic is not None:
            a_mime = magic_guess_custom.from_file(str(a))
            b_mime = magic_guess_custom.from_file(str(b))
            if a_mime != b_mime:
                return MIMEDiff(name, a_mime, b_mime)
            mime = a_mime
        else:
            warn("mime not specified: either specify it or install python-magic for a detailed diff")
    if mime is None:
        return PathDiff(name, eq=False, message=f"failed to determine MIME; tried libmagic: {magic is not None}")
    try:
        kernel = mime_dispatch[mime]
    except KeyError:
        return PathDiff(name, eq=False, message=f"unknown common MIME: {mime}")
    kwargs = {
        "min_ratio": min_ratio,
        "min_ratio_row": min_ratio_row,
        "max_cost": max_cost,
        "max_cost_row": max_cost_row,
    }
    if kernel in (diff_pd_csv, diff_pd_feather, diff_pd_parquet, diff_pd_excel):
        kwargs["table_drop_cols"] = table_drop_cols
        kwargs["table_sort"] = table_sort
        kwargs["align_col_data"] = align_col_data
    return kernel(a, b, name, **kwargs)
