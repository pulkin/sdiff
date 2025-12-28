import pytest

from sdiff.chunk import Diff, Chunk, Item, iter_chunks_important


def test_diff():
    diff = Diff(
        ratio=0,
        diffs=[
            Chunk("hello", "hi", False),
            Chunk("world", "world", True),
        ]
    )
    assert diff.get_a() == "helloworld"
    assert diff.get_b() == "hiworld"
    assert diff.get_inflated_ab() == (list("hellohiworld"), list("hellohiworld"))
    assert diff.with_data(list(range(10)), list(range(10, 17))) == Diff(
        ratio=10./17,
        diffs=[
            Chunk(data_a=[0, 1, 2, 3, 4], data_b=[10, 11], eq=False),
            Chunk(data_a=[5, 6, 7, 8, 9], data_b=[12, 13, 14, 15, 16], eq=True),
        ]
    )


def test_important():
    diff_345 = Diff(ratio=2 / 3, diffs=[
        Chunk(data_a=[3], data_b=[3], eq=True),
        Chunk(data_a=[4], data_b=[9], eq=False),
        Chunk(data_a=[5], data_b=[5], eq=True),
    ])
    chunks = [
        Chunk(data_a=[[0, 1, 2]], data_b=[[0, 1, 2]], eq=True),
        Chunk(data_a=[[3, 4, 5]], data_b=[[3, 9, 5]], eq=True, details=[diff_345]),
        Chunk(data_a=[[6, 7, 8]], data_b=[[6, 7, 8]], eq=True),
        Chunk(data_a=[[9, 10, 11]], data_b=[[19, 20, 21]], eq=False),
        Chunk(data_a=[[12, 13, 14], [15, 16, 17]], data_b=[[12, 13, 14], [15, 16, 17]], eq=True),
    ]
    assert list(iter_chunks_important(chunks)) == [
        1,
        Item(a=[3, 4, 5], b=[3, 9, 5], ix_a=1, ix_b=1, diff=diff_345),
        1,
        Item(a=[9, 10, 11], b=None, ix_a=3, ix_b=None),
        Item(a=None, b=[19, 20, 21], ix_a=None, ix_b=3),
        2,
    ]
    assert list(iter_chunks_important(chunks, context_size=1)) == [
        Item(a=[0, 1, 2], b=[0, 1, 2], ix_a=0, ix_b=0),
        Item(a=[3, 4, 5], b=[3, 9, 5], ix_a=1, ix_b=1, diff=diff_345),
        Item(a=[6, 7, 8], b=[6, 7, 8], ix_a=2, ix_b=2),
        Item(a=[9, 10, 11], b=None, ix_a=3, ix_b=None),
        Item(a=None, b=[19, 20, 21], ix_a=None, ix_b=3),
        Item(a=[12, 13, 14], b=[12, 13, 14], ix_a=4, ix_b=4),
        1,
    ]


@pytest.fixture
def chunks():
    return [
        Chunk(data_a="012", data_b="012", eq=True),
        Chunk(data_a="34", data_b="34", eq=True),
        Chunk(data_a="ab", data_b="c", eq=False),
        Chunk(data_a="5", data_b="5", eq=True),
        Chunk(data_a="6", data_b="6", eq=True),
        Chunk(data_a="d", data_b="ef", eq=False),
        Chunk(data_a="7890", data_b="7890", eq=True),
    ]


def test_coarse_0(chunks):
    result = Diff(0.42, chunks).get_coarse(1)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="01234", data_b="01234", eq=True),
        Chunk(data_a="ab", data_b="c", eq=False),
        Chunk(data_a="56", data_b="56", eq=True),
        Chunk(data_a="d", data_b="ef", eq=False),
        Chunk(data_a="7890", data_b="7890", eq=True),
    ]


def test_coarse_1(chunks):
    result = Diff(0.42, chunks).get_coarse(2)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="01234", data_b="01234", eq=True),
        Chunk(data_a="ab56d", data_b="c56ef", eq=False),
        Chunk(data_a="7890", data_b="7890", eq=True),
    ]


def test_coarse_2(chunks):
    result = Diff(0.42, chunks[1:]).get_coarse(2)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="34ab56d", data_b="34c56ef", eq=False),
        Chunk(data_a="7890", data_b="7890", eq=True),
    ]


def test_coarse_3(chunks):
    result = Diff(0.42, chunks[2:]).get_coarse(2)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="ab56d", data_b="c56ef", eq=False),
        Chunk(data_a="7890", data_b="7890", eq=True),
    ]


def test_coarse_4(chunks):
    result = Diff(0.42, chunks[:-1]).get_coarse(2)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="01234", data_b="01234", eq=True),
        Chunk(data_a="ab56d", data_b="c56ef", eq=False),
    ]


def test_coarse_5(chunks):
    result = Diff(0.42, chunks).get_coarse(100)
    assert result.ratio == 0.42
    assert result.diffs == [
        Chunk(data_a="01234ab56d7890", data_b="01234c56ef7890", eq=False),
    ]
