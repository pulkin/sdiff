import itertools
from typing import Any, Optional, Union
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import reduce, cached_property
from operator import add
from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkSignature:
    """
    Represents a chunk signature.

    Parameters
    ----------
    size_a
        Length of a sub-sequence in a.
    size_b
        Length of a sub-sequence in b.
    eq
        A flag indicating whether the two are considered
        equal.
    """
    size_a: int
    size_b: int
    eq: bool

    def __len__(self):
        if self.eq:
            return self.size_a
        else:
            return self.size_a + self.size_b

    @classmethod
    def aligned(cls, n: int) -> "ChunkSignature":
        return cls(size_a=n, size_b=n, eq=True)

    @classmethod
    def delta(cls, n: int, m: int) -> "ChunkSignature":
        return cls(size_a=n, size_b=m, eq=False)


@dataclass(frozen=True)
class Signature:
    """
    Represents a diff signature.

    Parameters
    ----------
    parts
        Signature constituents.
    """
    parts: Sequence[ChunkSignature]

    def __len__(self):
        return sum(len(i) for i in self.parts)

    @property
    def ratio(self):
        return (
            sum(((i.size_a + i.size_b) * i.eq) for i in self.parts) /
            sum((i.size_a + i.size_b) for i in self.parts)
        )

    @classmethod
    def aligned(cls, n: int) -> "Signature":
        if n == 0:
            return cls(tuple())
        return cls((ChunkSignature.aligned(n),))


@dataclass(frozen=True)
class Chunk:
    """
    Represents a chunk of two sequences being compared:
    either equal or not equal. A sequence of such chunks
    forms a diff.

    Parameters
    ----------
    data_a
        A chunk in the first sequence.
    data_b
        A chunk in the second sequence.
    eq
        A flag indicating whether the two are considered
        equal or a list of diffs specifying in which way
        pairs of items from data_a and data_b are different.
    """
    data_a: Sequence[Any]
    data_b: Sequence[Any]
    eq: Union[bool, Sequence[Union[bool, "Diff"]]]

    def to_string(
            self,
            prefix: str = "",
            uri_a: str = "a",
            uri_b: str = "b",
            offset_a: int = 0,
            offset_b: int = 0,
    ) -> str:
        eq = self.eq
        data_a = self.data_a
        data_b = self.data_b

        summary_uri_a = f"{uri_a}[{offset_a}:{offset_a + len(data_a)}]"
        summary_uri_b = f"{uri_b}[{offset_b}:{offset_b + len(data_b)}]"

        is_nested = isinstance(eq, Sequence)
        if is_nested:
            s = "≈"
        else:
            s = "=" if eq else "≠"
        base = f"{prefix}{summary_uri_a}{s}{summary_uri_b}: {repr(data_a)} {s} {repr(data_b)}"

        if not is_nested:
            return base

        # a sequence of aligned elements with some differences
        result = [base]
        for _i, (_eq, _a, _b) in enumerate(zip(eq, data_a, data_b)):
            result.append(_eq.to_string(
                prefix=prefix + "··",
                uri_a=f"{uri_a}[{offset_a + _i}]",
                uri_b=f"{uri_b}[{offset_b + _i}]",
            ))
        return "\n".join(result)

    @cached_property
    def signature(self) -> ChunkSignature:
        return ChunkSignature(
            size_a=len(self.data_a),
            size_b=len(self.data_b),
            eq=self.eq is not False,
        )

    def __add__(self, other: "Chunk") -> "Chunk":
        if not isinstance(other, Chunk):
            raise TypeError(f"cannot add {type(other)} to Chunk")
        if not isinstance(self.eq, bool) or not isinstance(other.eq, bool):
            raise TypeError("cannot add Chunks with nested diffs")
        return Chunk(
            data_a=self.data_a + other.data_a,
            data_b=self.data_b + other.data_b,
            eq=self.eq and other.eq,
        )


def iter_compressed_chunks(chunks: Iterable[Chunk]) -> Iterator[Chunk]:
    """
    Iterates through compressed chunks of differences.

    This method groups consecutive chunks based on equality criteria (`i.eq`)
    and yields each group as a single reduced chunk.

    Parameters
    ----------
    chunks
        Input chunks.

    Yields
    ------
    Each yielded value is a combined group of differences compressed into
    a single `Chunk` object.
    """
    for key, group in itertools.groupby(chunks, key=lambda i: i.eq):
        yield reduce(add, group)


def iter_coarse_chunks(chunks: Iterable[Chunk], consume_size: int) -> Iterable[Chunk]:
    """
    Iterates over chunks of differences and merges smaller equal chunks into
    bigger non-equal ones.

    Parameters
    ----------
    chunks
        Input chunks.

    consume_size : int
        The threshold size below which equal chunks are merged into non-equal ones.

    Yields
    ------
    A generator yielding bigger `Chunk` objects.
    """
    buffer = []
    for chunk in iter_compressed_chunks(chunks):
        if chunk.eq and len(chunk.data_a) > consume_size:
            if buffer:
                yield reduce(add, buffer)
            buffer = []
            yield chunk
        else:
            buffer.append(chunk)

    if buffer:
        yield reduce(add, buffer)


CrossHookType = Callable[[Any], Any]


@dataclass(frozen=True)
class Diff:
    """
    Represents a generic diff.

    Parameters
    ----------
    ratio
        The similarity ratio: a number between 0 and 1 telling
        the degree of the similarity in this diff. "0" typically
        means that there are no matches in this diff while "1"
        indicates a full alignment.
    diffs
        A list of diff chunks.
    """
    ratio: float
    diffs: Optional[list[Chunk]]

    @classmethod
    def from_signature(cls, sig: Signature, data_a, data_b) -> "Diff":
        """
        Prepares a diff given the diff signature and the two sequences.

        Parameters
        ----------
        sig
            The diff signature.
        data_a
        data_b
            The sequences to source data from.

        Returns
        -------
        The resulting diff with the provided signature and data from `a` and `b`.
        """
        diffs = []
        offset_a = offset_b = 0
        for part in sig.parts:
            diffs.append(Chunk(
                data_a=data_a[offset_a:(offset_a := offset_a + part.size_a)],
                data_b=data_b[offset_b:(offset_b := offset_b + part.size_b)],
                eq=part.eq,
            ))
        if offset_a != len(data_a):
            raise ValueError(f"{len(data_a)=}; expected: {offset_a}")
        if offset_b != len(data_b):
            raise ValueError(f"{len(data_b)=}; expected: {offset_b}")
        return Diff(ratio=sig.ratio, diffs=diffs)

    def __float__(self):
        return float(self.ratio)

    def __le__(self, other):
        return float(self) <= other

    def __lt__(self, other):
        return float(self) < other

    def __ge__(self, other):
        return float(self) >= other

    def __gt__(self, other):
        return float(self) > other

    def __bool__(self):
        if self.diffs is None:
            raise ValueError("no diff information available")
        if len(self.diffs) == 0:
            return True
        elif len(self.diffs) == 1:
            c, = self.diffs
            if c.eq is True:
                return True
            elif c.eq is False:
                return False
            else:
                raise ValueError(f"a nested diff cannot be cast to a bool; inner diff: {c.eq}")
        else:
            raise ValueError(f"a non-trivial diff cannot be cast to a bool ({len(self.diffs)} chunks)")

    def get_a(self):
        """
        Computes the first sequence.

        Returns
        -------
        The first sequence.
        """
        if self.diffs is None:
            raise ValueError("no diff data")
        return reduce(add, (i.data_a for i in self.diffs))

    def get_b(self):
        """
        Computes the second sequence.

        Returns
        -------
        The second sequence.
        """
        if self.diffs is None:
            raise ValueError("no diff data")
        return reduce(add, (i.data_b for i in self.diffs))

    def get_inflated_ab(self,
            hook_a_b: Optional[CrossHookType] = None,
            hook_b_a: Optional[CrossHookType] = None,
    ) -> tuple[list[Any], list[Any]]:
        """
        Computes a pair of aligned sequences by inflating sequence `a` to include the values from `b` and vice versa.

        Returns
        -------
        The two inflated sequences.
        """
        a = []
        b = []

        for chunk in self.diffs:
            a.append(chunk.data_a)
            if not chunk.eq:
                a.append(chunk.data_b if hook_a_b is None else type(chunk.data_a)(map(hook_a_b, chunk.data_b)))
                b.append(chunk.data_a if hook_b_a is None else type(chunk.data_b)(map(hook_b_a, chunk.data_a)))
            b.append(chunk.data_b)

        return reduce(add, a), reduce(add, b)

    def with_data(self, data_a, data_b) -> "Diff":
        """
        Creates a diff with the same signature but different data.

        Parameters
        ----------
        data_a
        data_b
            The new data.

        Returns
        -------
        A diff with the new data.
        """
        return self.from_signature(self.signature, data_a, data_b)

    def to_string(self, prefix: str = "", uri_a: str = "a", uri_b: str = "b") -> str:
        preamble = f"{prefix}{uri_a}≈{uri_b} (ratio={self.ratio:.4f})"
        if self.diffs is None:
            return preamble
        else:
            result = [preamble]
            offset_a = offset_b = 0
            for i in self.diffs:
                result.append(i.to_string(
                    prefix=prefix + "··",
                    uri_a=uri_a,
                    uri_b=uri_b,
                    offset_a=offset_a,
                    offset_b=offset_b,
                ))
                offset_a += len(i.data_a)
                offset_b += len(i.data_b)

            return "\n".join(result)

    @cached_property
    def signature(self) -> Signature:
        if self.diffs is None:
            raise ValueError("no diff data")
        return Signature(parts=tuple(i.signature for i in self.diffs))

    def iter_important(self, context_size: int = 0) -> Iterator[Union["Item", int]]:
        """
        Iterates over non-equal item pairs.

        Parameters
        ----------
        context_size
            The number of equal pairs to provide the context
            while yielding non-equal pairs.

        Yields
        ------
        Diff items and integers specifying the number of skipped
        item pairs in-between.
        """
        def _dummy():
            return
            yield

        def _head(data_a, counter_a, data_b, counter_b):
            for (i, a), (j, b) in zip(
                enumerate(data_a[:context_size], counter_a),
                enumerate(data_b[:context_size], counter_b),
            ):
                yield Item(a=a, b=b, ix_a=i, ix_b=j)

        def _tail(data_a, counter_a, data_b, counter_b, head_size):
            gap = len(data_a) - context_size - head_size
            if gap > 0:
                yield gap
            else:
                gap = 0
            gap += head_size
            for (i, a), (j, b) in zip(
                enumerate(data_a[gap:], counter_a + gap),
                enumerate(data_b[gap:], counter_b + gap),
            ):
                yield Item(a=a, b=b, ix_a=i, ix_b=j)

        context_tail = _dummy()
        counter_a = counter_b = 0

        for i_chunk, chunk in enumerate(self.diffs):

            if not isinstance(chunk.eq, Iterable):  # bool
                if chunk.eq:  # chunks are equal: take care of context
                    if i_chunk:  # this is NOT the beginning of text: yield context
                        yield from _head(chunk.data_a, counter_a, chunk.data_b, counter_b)
                    context_tail = _tail(chunk.data_a, counter_a, chunk.data_b, counter_b, bool(i_chunk) * context_size)

                else:  # chunks are not equal: yield them all
                    yield from context_tail
                    for i, a in enumerate(chunk.data_a, counter_a):
                        yield Item(a=a, b=None, ix_a=i, ix_b=None)
                    for j, b in enumerate(chunk.data_b, counter_b):
                        yield Item(a=None, b=b, ix_a=None, ix_b=j)

            else:  # chunks are aligned
                yield from context_tail
                for i, (a, b, diff) in enumerate(zip(chunk.data_a, chunk.data_b, chunk.eq)):
                    yield Item(a=a, b=b, ix_a=counter_a + i, ix_b=counter_b + i, diff=diff)

            counter_a += len(chunk.data_a)
            counter_b += len(chunk.data_b)

        leftover = sum(i if isinstance(i, int) else 1 for i in context_tail)
        if leftover:
            yield leftover

    def get_coarse(self, consume_size: int) -> "Diff":
        """
        Computes a coarse diff by merging smaller equal chunks into non-equal ones.

        Parameters
        ----------
        consume_size : int
         The threshold size below which equal chunks are merged into non-equal ones.

        Returns
        -------
        The resulting diff.
        """
        return Diff(ratio=self.ratio, diffs=list(iter_coarse_chunks(self.diffs, consume_size)))


@dataclass(frozen=True)
class Item:
    """
    Represents a diff item.

    Parameters
    ----------
    a
    b
        The two objects in diff.
    ix_a
    ix_b
        The corresponding sequence indexes.
    diff
        An optional diff between a and b.
    """
    a: Any
    b: Any
    ix_a: Optional[int]
    ix_b: Optional[int]
    diff: Optional[Diff] = None
