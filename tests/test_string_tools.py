import pytest

from sdiff.presentation.string_tools import align, iter_escape, visible_len

pure = "a very long colored string"
hello = f"\033[30m{pure}\033[m"
elli = "..."


def test_iter_escape():
    assert list(iter_escape(hello)) == [
        ("\033[30m", False),
        (pure, True),
        ("\033[m", False),
    ]


def test_vlen():
    assert visible_len(hello) == len(pure)


@pytest.mark.parametrize("n", list(range(3)))
def test_align_truncate_short(n):
    with pytest.raises(ValueError, match="ellipsis is too long"):
        align(hello, n, elli=elli)


@pytest.mark.parametrize("n", list(range(4, 8)))
def test_align_truncate_0(n):
    result = align(hello, n, elli=elli)
    assert result == "\033[30m" + pure[: n - len(elli)] + elli
    assert visible_len(result) == n


def test_align_truncate_1(n=30):
    result = align(hello, n, elli=elli)
    assert result == hello + " " * (n - len(pure))
    assert visible_len(result) == n
