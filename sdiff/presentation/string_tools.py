import re
from collections.abc import Iterator


def escape(s: str) -> str:
    return repr(s)[1:-1]


re_tformat = re.compile(
    r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
""",
    re.VERBOSE,
)  # https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python


def iter_escape(s: str) -> Iterator[tuple[str, bool]]:
    """
    Iterates over escape and non-escape sequences.

    Parameters
    ----------
    s
        String to process.

    Yields
    ------
    Pairs of substrings and bools telling whether the substring is a non-escape string.
    """
    prev = 0
    for i in re.finditer(re_tformat, s):
        start = i.start()
        if start != prev:
            yield s[prev:start], True
        end = i.end()
        yield s[start:end], False
        prev = end
    if prev != len(s):
        yield s[prev:], True


def visible_len(s: str) -> int:
    """
    String length, excluding escape sequences.

    Parameters
    ----------
    s
        The string to measure the length.

    Returns
    -------
    String length.
    """
    return sum(len(sub) * is_p for sub, is_p in iter_escape(s))


def align(s: str, n: int, elli: str = "…", fill: str = " ", justify="left") -> str:
    """
    Aligns a string towards a specific length. Truncates if the string is longer
    than the provided number and justifies it if the string is shorter.

    Parameters
    ----------
    s
        The string to align.
    n
        The desired string length.
    elli
        Suffix to add when truncating the string.
    fill
        A character to use for justifying.
    justify
        Text justify.

    Returns
    -------
    The resulting string of the desired length.
    """
    n_elli = n - visible_len(elli)
    if n_elli < 0:
        raise ValueError("ellipsis is too long")

    pos = 0

    for chunk, is_p in iter_escape(s):
        _l = len(chunk)

        if is_p:
            n -= _l

            # track where to stop accommodating ellipsis
            if n_elli > 0:
                pos += min(_l, n_elli)
            n_elli -= _l

        else:
            if n_elli >= 0:
                pos += _l

    if n >= 0:  # len(s) >= n
        filler = (fill * (n // len(fill) + 1))[:n]  # justify
        if justify == "left":
            return s + filler
        elif justify == "right":
            return filler + s
        elif justify == "center":
            n = len(filler) // 2
            return filler[:n] + s + filler[n:]
        else:
            raise ValueError(f"unknown {justify=}")
    else:
        return s[:pos] + elli  # append ellipsis
