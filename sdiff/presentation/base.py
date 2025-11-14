from dataclasses import dataclass, field
from sys import stdout
from io import TextIOBase
import os
from typing import Union, Any, Optional
from collections.abc import Sequence, Callable, Iterator
from itertools import groupby
import re
from html import escape as html_escape
from textwrap import dedent

from .string_tools import align, visible_len
from ..contextual.base import AnyDiff
from ..contextual.table import TableDiff
from ..contextual.text import TextDiff
from ..contextual.path import PathDiff, CompositeDiff, DeltaDiff, MIMEDiff
from ..chunk import Item, Diff, iter_chunks_verbose, iter_chunks_important


class TableBreak(str):
    pass


class TableHline(str):
    pass


@dataclass
class Table:
    column_mask: Sequence[bool]
    data: list[Union[tuple[str, ...], TableBreak, TableHline]] = field(default_factory=list)
    etc: str = "..."
    pre_str: Callable[[Any], str] = str
    """
    Represents a simple table.
    
    Parameters
    ----------
    column_mask
        Columns to display. The total number of columns
        is the length of this mask.
    data
        Column / cell data.
    etc
        A constant string to replace blocks of columns that
        are not displayed.
    pre_str
        A function pre-converting objects to str.
    """

    @property
    def row_len(self) -> int:
        """
        The number of rows displayed, including placeholders
        for rows that are skipped.
        """
        return sum(
            sum(1 for _ in group)
            if key
            else 1
            for key, group in groupby(self.column_mask)
        )

    def append_break(self, s: str):
        """
        Appends a break string.

        Parameters
        ----------
        s
            The break string.
        """
        self.data.append(TableBreak(s))

    def append_hline(self, s: str):
        """
        Appends a horizontal line.

        Parameters
        ----------
        s
            A printing symbol or sequence of symbols
            composing the line.
        """
        self.data.append(TableHline(s))

    def append_row(self, row: Sequence[object]):
        """
        Appends a table row. Additional elements in the
        row will be skipped.

        Parameters
        ----------
        row
            A sequence of objects to add as a row.
        """
        row_repr = []
        row_iter = iter(row)
        for key, group in groupby(self.column_mask):
            if key:
                for _ in group:
                    row_repr.append(self.pre_str(next(row_iter)))
            else:
                for _ in group:
                    next(row_iter)
                row_repr.append(self.etc)
        self.data.append(tuple(row_repr))

    def get_full_widths(self, m: int = 1) -> list[int]:
        """
        Computes actual row widths.

        Parameters
        ----------
        m
            Minimal column width.

        Returns
        -------
        A list of integers with row widths.
        """
        return [max(max(visible_len(d[i]), m) for d in self.data if type(d) is tuple) for i in range(self.row_len)]

    def compute(self, join: str, widths: Optional[list[int]] = None, elli: str = "…") -> Iterator[str]:
        """
        Computes the table.

        Parameters
        ----------
        join
            A string for joining cells.
        widths
            Row widths to use.
        elli
            Ellipsis.

        Returns
        -------
        A sequence of table rows.
        """
        justify = "center"
        for i in self.data:
            if isinstance(i, tuple):
                if widths is None:
                    yield join.join(i)
                else:
                    yield join.join(align(s, n, elli=elli, justify=justify) for s, n in zip(i, widths))
                    justify = "right"
            elif isinstance(i, TableBreak):
                yield str(i)
            elif isinstance(i, TableHline):
                if widths is None:
                    yield i
                else:
                    yield join.join((i * (w // len(i) + 1))[:w] for w in widths)
            else:
                raise ValueError(f"unknown row type: {type(i)}")


@dataclass
class TextFormats:
    header: str = "{header}"
    textwrap_start: str = ""
    textwrap_end: str = ""
    del_entry: str = "DEL {path_key}"
    new_entry: str = "NEW {path_key}"
    mime_entry: str = "MIME {path_key} {path_a} ≠ {path_b}"
    same_entry: str = "same {path_key}"
    skip_equal: str = "({n} lines match)"
    line_ctx: str = "  {line}"
    line_add: str = "> {line}"
    line_rm: str = "< {line}"
    line_aligned: str = "≈ {line}"
    line_aligned_add: str = line_add
    line_aligned_rm: str = line_rm
    block_spacer: str = "---\n"
    chunk_add: str = "+++{chunk}+++"
    chunk_rm: str = "---{chunk}---"

    hello: str = ""
    goodbye: str = ""

    @staticmethod
    def escape(s: str) -> str:
        if s.endswith("\n"):
            s, x = s[:-1], s[-1:]
        else:
            x = ""
        return repr(s)[1:-1] + x


@dataclass
class MarkdownTextFormats(TextFormats):
    header: str = "{header}\n"
    textwrap_start: str = "~~~text\n"
    textwrap_end: str = "~~~\n"
    del_entry: str = "DEL {path_key}\n"
    new_entry: str = "NEW {path_key}\n"
    mime_entry: str = "MIME {path_key} {path_a} ≠ {path_b}\n"
    same_entry: str = "same {path_key}\n"
    skip_equal: str = "({n} lines match)"
    line_ctx: str = "  {line}"
    line_add: str = "> {line}"
    line_rm: str = "< {line}"
    line_aligned: str = "≈ {line}"
    line_aligned_add: str = line_add
    line_aligned_rm: str = line_rm
    block_spacer: str = "---\n"
    chunk_add: str = "+++{chunk}+++"
    chunk_rm: str = "---{chunk}---"

    hello: str = ""
    goodbye: str = ""

    @staticmethod
    def escape(s: str) -> str:
        return re.sub(f'(~~~)', r'[triple ~]', s)


_tformat = "\033[%dm%%s\033[0m"


tf_strike = _tformat % 9
tf_black = _tformat % 30
tf_red = _tformat % 31
tf_green = _tformat % 32
tf_on_red = _tformat % 41
tf_on_green = _tformat % 42
tf_on_light_grey = _tformat % 47
tf_grey = _tformat % 90
tf_on_white = _tformat % 107


@dataclass
class TermTextFormats(TextFormats):
    header: str = (tf_on_light_grey % tf_black)[:-4] % "{header}"
    textwrap_start: str = ""
    textwrap_end: str = ""
    del_entry: str = f"{tf_red % 'DEL'} {{path_key}}"
    new_entry: str = f"{tf_green % 'NEW'} {{path_key}}"
    mime_entry: str = "MIME {path_key} {path_a} ≠ {path_b}"
    same_entry: str = "same {path_key}"
    skip_equal: str = tf_grey % "({n} lines match)"
    line_ctx: str = tf_grey % "  {line}"
    line_add: str = tf_green % "> {line}"
    line_rm: str = tf_red % "< {line}"
    line_aligned: str = "≈ {line}"
    line_aligned_add: str = "> {line}"
    line_aligned_rm: str = "< {line}"
    block_spacer: str = ""
    chunk_add: str = (tf_on_green % tf_black)[:-4] % "{chunk}"
    chunk_rm: str = (tf_on_red % tf_black)[:-4] % "{chunk}"

    hello: str = ""
    goodbye: str = ""


@dataclass
class HTMLTextFormats(TextFormats):
    header: str = "<h4>{header}</h4>"
    textwrap_start: str = "<pre>"
    textwrap_end: str = "</pre>"
    del_entry: str = "<h4>DEL {path_key}</h4>"
    new_entry: str = "<h4>NEW {path_key}</h4>"
    mime_entry: str = "<h4>MIME {path_key} {path_a} ≠ {path_b}</h4>"
    same_entry: str = "<h4>same {path_key}</h4>"
    skip_equal: str = "({n} lines match)"
    line_ctx: str = "  {line}"
    line_add: str = "<span class=\"diff-add\">&gt; {line}</span>"
    line_rm: str = "<span class=\"diff-rm\">&lt; {line}</span>"
    line_aligned: str = "<span class=\"diff-highlight\">≈ {line}</span>"
    line_aligned_add: str = "<span class=\"diff-add-aligned\">&gt; {line}</span>"
    line_aligned_rm: str = "<span class=\"diff-rm-aligned\">&lt; {line}</span>"
    block_spacer: str = ""
    chunk_add: str = "<span class=\"diff-add\">{chunk}</span>"
    chunk_rm: str = "<span class=\"diff-rm\">{chunk}</span>"

    hello: str = dedent("""
    <!DOCTYPE html><html><head>
    <meta charset=\"UTF-8\">
    <link rel="stylesheet" href="https://unpkg.com/mvp.css"> 
    <style>
      .diff-rm {
        background-color: #B8405E;
        color: #F4E1E6;
        font-weight: bold;
      }
      .diff-add {
        background-color: #2EB086;
        color: #CEF2E7;
        font-weight: bold;
      }
      .diff-rm-aligned {
        background-color: #F4E1E6;
      }
      .diff-add-aligned {
        background-color: #CEF2E7;
      }
      td.diff-pair-top, td.diff-pair-bottom {
        border : 0.3em dotted black;
        box-shadow: 2px 2px 15px -2px grey;
      }
      td.diff-pair-top {
        border-bottom: none;
      }
      td.diff-pair-bottom {
        border-top: none;
      }
      .diff-highlight {
        background-color: #EEE6CE;
        display: block;
      }
      .diff-context {
        color: grey;
      }
      body {
        font-family: monospace;
      }
      table {
        border-collapse: collapse;
      }
      table td {
        text-align: right;
      }
      table td, table th, table tr {
        padding: 0.2rem 0.8rem;
      }
      table tr td:nth-child(2) {
        border-right: 1px dashed grey;
      }
    </style>
    </head><body>\n""")
    goodbye: str = "</body></html>\n"

    @staticmethod
    def escape(s: str) -> str:
        return html_escape(s)


@dataclass
class TableFormats:
    align: bool = True

    table_head: str = ""
    table_tail: str = ""

    skip_equal: str = "({n} row(s) match)"

    column_plain: str = "{column}"
    column_add: str = "+{column}"
    column_rm: str = "-{column}"
    column_both: str = "---{column_a}>>>{column_b}+++"

    ix_row_header: str = ""
    ix_row_none: str = ""
    ix_row_context_one: str = "{i}"
    ix_row_context_both: str = "{i_a}A{i_b}B"
    ix_row_same: str = "{i}"
    ix_row_a: str = "{i}A"
    ix_row_b: str = "{i}B"

    data_row_none: str = ""
    data_row_context: str = "{chunk}"
    data_row_same: str = "{chunk}"
    data_row_a: str = "---{chunk}---"
    data_row_b: str = "+++{chunk}+++"
    data_row_xa: str = data_row_a
    data_row_xb: str = data_row_b

    row_head: str = ""
    row_spacer: str = " "
    row_tail: str = ""

    hline: str = "-"
    elli: str = "…"


@dataclass
class TermTableFormats(TableFormats):
    align: bool = True

    table_head: str = ""
    table_tail: str = ""

    skip_equal: str = tf_grey % "({n} row(s) match)"

    column_plain: str = "{column}"
    column_add: str = tf_green % "{column}"
    column_rm: str = tf_red % "{column}"
    column_both: str = f"{tf_red % '{column_a}'}{tf_green % '{column_b}'}"

    ix_row_header: str = ""
    ix_row_none: str = ""
    ix_row_context_one: str = tf_grey % "{i}"
    ix_row_context_both: str = tf_grey % "{i_a}A{i_b}B"
    ix_row_same: str = "{i}"
    ix_row_a: str = tf_red % "{i}"
    ix_row_b: str = tf_green % "{i}"

    data_row_none: str = ""
    data_row_context: str = tf_grey % "{chunk}"
    data_row_same: str = "{chunk}"
    data_row_a: str = tf_red % "{chunk}"
    data_row_b: str = tf_green % "{chunk}"
    data_row_xa: str = data_row_a
    data_row_xb: str = data_row_b

    row_head: str = ""
    row_spacer: str = " "
    row_tail: str = ""

    hline: str = ""
    elli: str = "…\033[0m"


@dataclass
class MarkdownTableFormats(TableFormats):
    align: bool = True

    table_head: str = ""
    table_tail: str = ""

    skip_equal: str = "({n} row(s) match)"

    column_plain: str = "{column}"
    column_add: str = "*{column}*"
    column_rm: str = "~~{column}~~"
    column_both: str = "~~{column_a}~~*{column_b}*"

    ix_row_header: str = ""
    ix_row_none: str = ""
    ix_row_context_one: str = "{i}"
    ix_row_context_both: str = "{i_a}A{i_b}B"
    ix_row_same: str = "{i}"
    ix_row_a: str = "~~{i}~~"
    ix_row_b: str = "*{i}*"

    data_row_none: str = ""
    data_row_context: str = "{chunk}"
    data_row_same: str = "{chunk}"
    data_row_a: str = "~~{chunk}~~"
    data_row_b: str = "*{chunk}*"
    data_row_xa: str = data_row_a
    data_row_xb: str = data_row_b

    row_head: str = "| "
    row_spacer: str = " | "
    row_tail: str = " |"

    hline: str = "-"
    elli: str = "…"


@dataclass
class HTMLTableFormats(TableFormats):
    align: bool = False

    table_head: str = "<table>"
    table_tail: str = "</table>"

    skip_equal: str = "<td colspan=\"100%\" class=\"diff-context\">({n} row(s) match)</td>"

    column_plain: str = "<th>{column}</th>"
    column_add: str = "<th class=\"diff-add\">{column}</th>"
    column_rm: str = "<th class=\"diff-rm\">{column}</th>"
    column_both: str = "<th><span class=\"diff-rm\">{column_a}</span><span class=\"diff-add\">{column_b}</span></th>"

    ix_row_header: str = "<td class=\"diff-context\">A</td><td class=\"diff-context\">B</td>"
    ix_row_none: str = "<td></td><td></td>"
    ix_row_context_one: str = "<td class=\"diff-context\">{i}</td><td></td>"
    ix_row_context_both: str = "<td class=\"diff-context\">{i_a}</td><td class=\"diff-context\">{i_b}</td>"
    ix_row_same: str = "<td>{i}</td><td></td>"
    ix_row_a: str = "<td class=\"diff-rm\">{i}</td><td></td>"
    ix_row_b: str = "<td></td><td class=\"diff-add\">{i}</td>"

    data_row_none: str = "<td></td>"
    data_row_context: str = "<td class=\"diff-context\">{chunk}</td>"
    data_row_same: str = "<td>{chunk}</td>"
    data_row_a: str = "<td class=\"diff-rm\">{chunk}</td>"
    data_row_b: str = "<td class=\"diff-add\">{chunk}</td>"
    data_row_xa: str = "<td class=\"diff-rm diff-pair-top\">{chunk}</td>"
    data_row_xb: str = "<td class=\"diff-add diff-pair-bottom\">{chunk}</td>"

    row_head: str = "<tr>"
    row_spacer: str = ""
    row_tail: str = "</tr>"

    hline: str = ""
    elli: str = "…"


@dataclass
class AbstractTextPrinter:
    printer: TextIOBase = stdout
    verbosity: int = 0
    width: int = 0
    """
    Diff printer.

    Parameters
    ----------
    printer
        The text printer.
    verbosity
        The verbosity level.
    width
        Total screen width.
    """

    def __post_init__(self):
        if not self.width:
            try:
                self.width = os.get_terminal_size(self.printer.fileno()).columns
            except (AttributeError, ValueError, OSError):
                self.width = 80

    def print_hello(self):
        pass

    def print_goodbye(self):
        pass

    def print_diff(self, diff: AnyDiff):
        """
        Prints diff.

        Parameters
        ----------
        diff
            A diff to process.
        """
        if diff.is_eq():
            if self.verbosity >= 2:
                self.print_equal(diff)
        elif isinstance(diff, TextDiff):
            self.print_text(diff)
        elif isinstance(diff, TableDiff):
            self.print_table(diff)
        elif isinstance(diff, PathDiff):
            self.print_path(diff)
        elif isinstance(diff, DeltaDiff):
            self.print_delta(diff)
        elif isinstance(diff, MIMEDiff):
            self.print_mime(diff)
        elif isinstance(diff, CompositeDiff):
            for _d in diff.items:
                self.print_diff(_d)
        else:
            raise NotImplementedError(f"unknown diff: {diff}")

    def print_equal(self, diff: AnyDiff):
        raise NotImplementedError

    def print_path(self, diff: PathDiff):
        raise NotImplementedError

    def print_delta(self, diff: DeltaDiff):
        raise NotImplementedError

    def print_mime(self, diff: MIMEDiff):
        raise NotImplementedError

    def print_text(self, diff: TextDiff):
        raise NotImplementedError

    def print_table(self, diff: TableDiff):
        raise NotImplementedError


def _format_text_line(diff: Diff, a_fmt: Optional[str], b_fmt: Optional[str]) -> str:
    line_parts = []
    for chunk in diff.diffs:
        if chunk.eq:
            line_parts.append(chunk.data_a)
        else:
            if chunk.data_a and a_fmt is not None:
                line_parts.append(a_fmt.format(chunk=chunk.data_a))
            if chunk.data_b and b_fmt is not None:
                line_parts.append(b_fmt.format(chunk=chunk.data_b))
    return "".join(line_parts)


@dataclass
class TextPrinter(AbstractTextPrinter):
    context_size: int = 2
    text_split_aligned: bool = False
    table_collapse_columns: bool = False
    table_formats: TableFormats = field(default_factory=TableFormats)
    text_formats: TextFormats = field(default_factory=TextFormats)
    """
    A simple diff printer.
    
    Parameters
    ----------
    printer
        The text printer.
    verbosity
        The verbosity level.
    width
        Total screen width.
    context_size
        The number of non-diff rows to surround diffs with.
    text_split_aligned
        If True, splits aligned lines into added and removed lines.
    table_collapse_columns
        If True, collapses table columns that do not contain diffs.
    table_formats
        Table formats.
    text_format
        Text formats.
    """

    def print_hello(self):
        self.printer.write(self.text_formats.hello)

    def print_goodbye(self):
        self.printer.write(self.text_formats.goodbye)

    def print_equal(self, diff: AnyDiff):
        """
        Prints equal (empty) diff.

        Parameters
        ----------
        diff
            A diff to process.
        """
        p = self.printer.write
        add = ""
        if self.verbosity >= 1 and isinstance(diff, PathDiff) and diff.message is not None:
            add = f" ({diff.message})"
        p(self.text_formats.same_entry.format(path_key=f"{diff.name} -- {diff.__class__.__name__}{add}\n"))

    def print_header(self, diff: AnyDiff):
        """
        Prints diff header.

        Parameters
        ----------
        diff
            A diff to process.
        """
        p = self.printer.write
        v = self.verbosity

        p(self.text_formats.header.format(header=f"comparing {diff.name}"))
        if v >= 1:
            if isinstance(diff, (TableDiff, TextDiff)):
                p(f" (ratio={diff.data.ratio:.4f})")
        if v >= 2:
            if isinstance(diff, TableDiff):
                p(f"\n  aligned ratio={diff.data.aligned_ratio:.4f}")
                for name, orig, inflated in [
                    ("a", diff.data.a_shape, diff.data.a.shape),
                    ("b", diff.data.b_shape, diff.data.b.shape),
                ]:
                    n = orig[0] * orig[1]
                    if n:
                        x = f"{inflated[0] * inflated[1] / n:.2f}"
                    else:
                        x = "∞"
                    p(f"\n  {name}.shape={orig} -> {inflated} x{x}")
        p("\n")

    def print_path(self, diff: PathDiff):
        """
        Print a path diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        p = self.printer.write
        if diff.eq:
            self.print_equal(diff)
        else:
            p(f"{diff.name} are not equal")
            if self.verbosity >= 1 and diff.message is not None:
                p(f" ({diff.message})")
            p("\n")

    def print_delta(self, diff: DeltaDiff):
        """
        Print a delta diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        fmt = self.text_formats.del_entry if diff.exist_a else self.text_formats.new_entry
        self.printer.write(f"{fmt.format(path_key=diff.name)}\n")

    def print_mime(self, diff: MIMEDiff):
        """
        Print a MIME diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        self.printer.write(f"{self.text_formats.mime_entry.format(path_key=diff.name, path_a=diff.mime_a, path_b=diff.mime_b)}\n")

    def print_text(self, diff: TextDiff):
        """
        Prints a text diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        self.print_header(diff)
        p = self.printer.write
        e = self.text_formats.escape
        formats = {
            (True, False, False): self.text_formats.line_rm,
            (False, True, False): self.text_formats.line_add,
            (True, True, False): self.text_formats.line_ctx,
            (True, True, True): self.text_formats.line_aligned,
        }

        separator = False
        p(self.text_formats.textwrap_start)
        for is_skip, group in groupby(iter_chunks_important(diff.data.diffs, context_size=self.context_size), lambda i: isinstance(i, int)):
            if is_skip:
                for i in group:
                    p(self.text_formats.skip_equal.format(n=i) + "\n")
                    separator = False
            else:
                for key, group_2 in groupby(group, lambda i: (i.a is not None, i.b is not None, i.diff is not None)):
                    if separator:
                        p(self.text_formats.block_spacer)
                    separator = True
                    fmt = formats[key]
                    for i in group_2:
                        if i.a is None and i.b is not None:  # addition
                            p(fmt.format(line=e(i.b)))

                        elif i.b is None and i.a is not None:  # removal
                            p(fmt.format(line=e(i.a)))

                        elif i.diff is None:  # context
                            p(fmt.format(line=e(i.a)))

                        else:  # inline diff
                            if not self.text_split_aligned:
                                p(fmt.format(line=_format_text_line(
                                    i.diff,
                                    self.text_formats.chunk_rm,
                                    self.text_formats.chunk_add
                                )))
                            else:
                                for fmt_line, fmt_rm, fmt_add in [
                                    (self.text_formats.line_aligned_rm, self.text_formats.chunk_rm, None),
                                    (self.text_formats.line_aligned_add, None, self.text_formats.chunk_add),
                                ]:
                                    p(fmt_line.format(line=_format_text_line(i.diff, fmt_rm, fmt_add)))
        p(self.text_formats.textwrap_end)

    def print_table(self, diff: TableDiff):
        """
        Prints a table diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        self.print_header(diff)
        self.printer.write(self.table_formats.table_head)

        # hide columns (optionally)
        if self.table_collapse_columns:
            show_col = [True, *~diff.data.eq.all(axis=0)]
            if diff.columns is not None:
                show_col = [True, *(
                    i or col_a != col_b
                    for i, col_a, col_b in zip(show_col[1:], diff.columns.a, diff.columns.b)
                )]
        else:
            show_col = [True] * (diff.data.eq.shape[1] + 1)
        table = Table(column_mask=show_col)

        # print column names
        if diff.columns is not None:
            row = [self.table_formats.ix_row_header.format(col_ix=0)]
            for col_a, col_b in zip(diff.columns.a, diff.columns.b):
                col_ix = len(row)
                if col_a == col_b:
                    col = self.table_formats.column_plain.format(column=col_a, col_ix=col_ix)
                elif not col_a:
                    col = self.table_formats.column_add.format(column=col_b, col_ix=col_ix)
                elif not col_b:
                    col = self.table_formats.column_rm.format(column=col_a, col_ix=col_ix)
                else:
                    col = self.table_formats.column_both.format(column_a=col_a, column_b=col_b, col_ix_a=col_ix, col_ix_b=col_ix + 1)
                row.append(col)
            table.append_row(row)

        if self.table_formats.hline:
            table.append_hline(self.table_formats.hline)

        # print table data
        for i in iter_chunks_important(diff.data.to_plain().diffs, context_size=self.context_size):
            if isinstance(i, int):
                table.append_break(self.table_formats.skip_equal.format(n=i))
            elif isinstance(i, Item):

                if i.a is None:  # addition
                    table.append_row([self.table_formats.ix_row_b.format(i=i.ix_b), *(self.table_formats.data_row_b.format(chunk=s) for s in i.b)])

                elif i.b is None:  # removal
                    table.append_row([self.table_formats.ix_row_a.format(i=i.ix_a), *(self.table_formats.data_row_a.format(chunk=s) for s in i.a)])

                elif i.diff is None:  # context
                    if i.ix_a == i.ix_b:
                        code = self.table_formats.ix_row_context_one.format(i=i.ix_a)
                    else:
                        code = self.table_formats.ix_row_context_both.format(i_a=i.ix_a, i_b=i.ix_b)
                    table.append_row([code, *(self.table_formats.data_row_context.format(chunk=s) for s in i.a)])

                else:  # inline diff
                    row_a = []
                    row_b = []
                    any_row_b = False
                    if i.ix_a != i.ix_b:
                        row_a.append(self.table_formats.ix_row_a.format(i=i.ix_a, col_ix=len(row_a)))
                        row_b.append(self.table_formats.ix_row_b.format(i=i.ix_b, col_ix=len(row_b)))
                        any_row_b = True
                    else:
                        row_a.append(self.table_formats.ix_row_same.format(i=i.ix_a, col_ix=len(row_a)))
                        row_b.append(self.table_formats.ix_row_none.format(col_ix=len(row_b)))
                    for a, b, eq in zip(i.a, i.b, i.diff):
                        if eq:
                            row_a.append(self.table_formats.data_row_same.format(chunk=a, col_ix=len(row_a)))
                            row_b.append(self.table_formats.data_row_none.format(col_ix=len(row_b)))
                        else:
                            if bool(a) and bool(b):
                                row_a.append(self.table_formats.data_row_xa.format(chunk=a, col_ix=len(row_a)))
                                row_b.append(self.table_formats.data_row_xb.format(chunk=b, col_ix=len(row_b)))
                                any_row_b = True
                            else:
                                if a:
                                    row_a.append(self.table_formats.data_row_a.format(chunk=a, col_ix=len(row_a)))
                                elif b:
                                    row_a.append(self.table_formats.data_row_b.format(chunk=b, col_ix=len(row_a)))
                                else:
                                    row_a.append(self.table_formats.data_row_none.format(col_ix=len(row_a)))
                                row_b.append(self.table_formats.data_row_none.format(col_ix=len(row_b)))
                    table.append_row(row_a)
                    if any_row_b:
                        table.append_row(row_b)

        widths = None
        # trim table width
        if self.width and self.table_formats.align:
            min_width = 1
            widths = table.get_full_widths(min_width)
            head_len = visible_len(self.table_formats.row_head)
            spacer_len = visible_len(self.table_formats.row_spacer)
            tail_len = visible_len(self.table_formats.row_tail)
            # adjustable width excludes spacer, head, tail, index column and minimal width
            adj_width = max(0, self.width - head_len - tail_len - (len(widths) - 1) * (min_width + spacer_len) - widths[0])

            cs = [0]  # np.cumsum
            s = 0
            for w in widths[1:]:
                s += w - min_width
                cs.append(s)
            if s > adj_width:
                ratio = adj_width / s
                cs = [int(i * ratio) for i in cs]
            widths = [widths[0], *(j - i + min_width for i, j in zip(cs[:-1], cs[1:]))]

        for row in table.compute(self.table_formats.row_spacer, widths=widths, elli=self.table_formats.elli):
            self.printer.write(self.table_formats.row_head + row + self.table_formats.row_tail + "\n")

        self.printer.write(self.table_formats.table_tail)


@dataclass
class TextSummaryFormats:
    ratio_fmt: str = "{:.4f}"
    ratio_non_fmt: str = "{:<6}"
    n_equal_fmt: str = "={:<7d}"
    n_equal_non_fmt: str = "{:<8}"
    n_neq_fmt: str = "≠{:<7d}"
    n_neq_non_fmt: str = "{:<8}"
    n_aligned_fmt: str = "≈{:<7d}"
    n_aligned_non_fmt: str = "{:<8}"
    sep = " "


@dataclass
class SummaryTextPrinter(AbstractTextPrinter):
    formats: TextSummaryFormats = field(default_factory=TextSummaryFormats)
    """
    A summary diff printer.

    Parameters
    ----------
    printer
        The text printer.
    verbosity
        The verbosity level.
    width
        Total screen width.
    formats
        Text formats.
    """

    @property
    def _empty_fmt(self) -> str:
        f = self.formats
        return f"{f.ratio_non_fmt}{f.sep}{f.n_equal_non_fmt}{f.sep}{f.n_aligned_non_fmt}{f.sep}{f.n_neq_non_fmt}{f.sep}{{}}"

    @property
    def _ratio_fmt(self) -> str:
        f = self.formats
        return f"{f.ratio_fmt}{f.sep}{f.n_equal_non_fmt}{f.sep}{f.n_aligned_non_fmt}{f.sep}{f.n_neq_non_fmt}{f.sep}{{}}"

    @property
    def _full_fmt(self) -> str:
        f = self.formats
        return f"{f.ratio_fmt}{f.sep}{f.n_equal_fmt}{f.sep}{f.n_aligned_fmt}{f.sep}{f.n_neq_fmt}{f.sep}{{}}"

    def print_equal(self, diff: AnyDiff):
        """
        Prints equal (empty) diff.

        Parameters
        ----------
        diff
            A diff to process.
        """
        self.printer.write(self._empty_fmt.format("1", "", "", "", f"{diff.name} match\n"))

    def print_path(self, diff: PathDiff):
        """
        Print a path diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        p = self.printer.write
        if diff.eq:
            self.print_equal(diff)
        else:
            p(self._empty_fmt.format("0", "", "", "", f"{diff.name} do not match"))
            if self.verbosity >= 1 and diff.message is not None:
                p(f" ({diff.message})")
            p("\n")

    def print_delta(self, diff: DeltaDiff):
        """
        Print a delta diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        self.printer.write(self._empty_fmt.format('DEL' if diff.exist_a else 'NEW', "", "", "", f"{diff.name}\n"))

    def print_mime(self, diff: MIMEDiff):
        """
        Print a MIME diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        self.printer.write(self._empty_fmt.format("MIME", "", "", "", f"{diff.name} {diff.mime_a} ≠ {diff.mime_b}\n"))

    def print_text(self, diff: TextDiff):
        """
        Prints a text diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        if diff.data.diffs is not None:
            n_eq = n_al = n_ne = 0
            for is_eq, is_exact, chunk in iter_chunks_verbose(diff.data.diffs):
                s = len(chunk.data_a)  # TODO: include data_b?
                n_eq += s * is_eq * is_exact
                n_al += s * is_eq * (not is_exact)
                n_ne += s * (not is_eq)
            self.printer.write(self._full_fmt.format(diff.data.ratio, n_eq, n_al, n_ne, f"{diff.name}\n"))
        else:
            self.printer.write(self._ratio_fmt.format(diff.data.ratio, "", "", "", f"{diff.name}\n"))

    def print_table(self, diff: TableDiff):
        """
        Prints a table diff.

        Parameters
        ----------
        diff
            The diff to print.
        """
        n_eq = diff.data.eq.all(axis=1).sum()
        n_al = diff.data.eq.any(axis=1).sum() - n_eq
        n_ne = (~diff.data.eq).all(axis=1).sum()
        self.printer.write(self._full_fmt.format(diff.data.ratio, n_eq, n_al, n_ne, f"{diff.name}\n"))
