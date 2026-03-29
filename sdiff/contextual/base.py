from dataclasses import dataclass
from collections import defaultdict
from collections.abc import Mapping
from functools import wraps
import time
from typing import Optional


@dataclass
class AnyDiff:
    name: str
    """
    An empty top-level diff type.
    
    Parameters
    ----------
    name
        A name this diff belongs to.
    """

    def is_eq(self) -> bool:
        """
        Checks if diff is trivial and the two objects compared are equal.

        Returns
        -------
        True if equal.
        """
        raise NotImplementedError


def profile(name: str):
    def wrapper(f):
        @wraps(f)
        def result(*args, **kwargs):
            t = time.time()
            rtn = f(*args, **kwargs)
            t = time.time() - t
            rtn.stats[name] = t - rtn.stats.get("_time", 0)
            rtn.stats["_time"] = t
            return rtn

        return result

    return wrapper


def add_stats(
    stats: Mapping[str, float], base_stats: Optional[defaultdict[str, float]]
) -> defaultdict[str, float]:
    if base_stats is None:
        base_stats = defaultdict(float)
    for k, v in stats.items():
        base_stats[k] += v
    return base_stats
