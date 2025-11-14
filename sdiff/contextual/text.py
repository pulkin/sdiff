from dataclasses import dataclass, field, replace
from collections.abc import Mapping, Sequence
from numbers import Number

from .base import AnyDiff, profile
from ..sequence import MAX_COST, MIN_RATIO, diff_nested
from ..chunk import Diff, Chunk


@dataclass
class TextDiff(AnyDiff):
    data: Diff
    stats: Mapping[str, Number] = field(default_factory=dict, compare=False)
    """
    A text diff.

    Parameters
    ----------
    name
        A name this diff belongs to.
    data
        Diff data.
    stats
        Stats associated with this diff.
    """

    def is_eq(self) -> bool:
        return all((i.eq is True and i.details is None) for i in self.data.diffs)


@profile("text comparison")
def diff(
        a: Sequence[str],
        b: Sequence[str],
        name: str,
        min_ratio: float = MIN_RATIO,
        min_ratio_row: float = MIN_RATIO,
        max_cost: int = MAX_COST,
        max_cost_row: int = MAX_COST,
        coarse: int = 4,
) -> TextDiff:
    """
    Computes a diff between two texts.

    Parameters
    ----------
    a
        The first text.
    b
        The second text.
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
    coarse
        Removes all equal text fragments below the given size.

    Returns
    ------
    The text diff.
    """
    raw_diff = diff_nested(
        a=a,
        b=b,
        min_ratio=(min_ratio, min_ratio_row),
        max_cost=(max_cost, max_cost_row),
        max_depth=2,
    )
    if coarse:
        new_diffs = []
        for i in raw_diff.diffs:
            if i.eq:
                new_details = []
                for j in i.details:
                    post_last = None
                    if len(j.diffs) > 0:
                        last = j.diffs[-1]
                        if last.eq:
                            a = last.data_a
                            b = last.data_b
                            last = replace(last, data_a=a.rstrip("\n"), data_b=b.rstrip("\n"))
                            if last.data_a != a:
                                post_last = Chunk(eq=True, data_a=a[len(last.data_a):], data_b=b[len(last.data_b):])
                        j = replace(j, diffs=j.diffs[:-1] + [last])
                    coarse_diff = j.get_coarse(coarse)
                    if post_last is not None:
                        coarse_diff = replace(coarse_diff, diffs=coarse_diff.diffs + [post_last])
                    new_details.append(coarse_diff)
                new_diffs.append(replace(i, details=new_details))
            else:
                new_diffs.append(i)
        raw_diff = replace(raw_diff, diffs=new_diffs)
    return TextDiff(
        name=name,
        data=raw_diff,
    )
