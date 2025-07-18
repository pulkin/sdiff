from pathlib import Path

import pytest

from .util import git_self_extract, process2text, sync_contents

cases = Path(__file__).parent / "cases"


@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["-v"], "default-v.txt"),
    (["-vv"], "default-vv.txt"),
    (["--format", "color"], "color.txt"),
    (["--format", "md"], "markdown.md"),
    (["--format", "summary"], "summary.txt"),
    (["--format", "html"], "page.html"),
    (["--exclude", "tests/", "--format", "summary"], "exclude-1.txt"),
    (["--exclude", "tests/test_presentation/cases/co2_emissions/", "--format", "summary"], "exclude-2.txt"),
    (["--reverse", "--format", "summary"], "reverse.txt"),
    (["--cherry-pick", "diff.feather.txt"], "cherry-pick.txt"),
    (["--shallow"], "shallow.txt"),
])
def test_git(tmp_path, test_diff_renders, args, name):
    git_self_extract("0c197f2cdb0bf8c0ca95e76a837296fbebad436d", a := tmp_path / "a")
    git_self_extract("e807d333433209f9328decc8290d40c270d832cd", b := tmp_path / "b")
    code, text = process2text([str(a), str(b), *args])
    sync_contents(cases / f"git/diff-{name}", text, test_diff_renders)
    assert code is True

    # test pool
    code, text = process2text([str(a), str(b), "--pool", "2", *args])
    sync_contents(cases / f"git/diff-{name}", text, True)
    assert code is True

    # test pool not ordered
    code, text = process2text([str(a), str(b), "--pool", "2", *args], sort=False)
    sync_contents(cases / f"git/diff-{name}", text, True, eq=lambda i, j: set(i) == set(j))
    assert code is True


@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["-vv"], "verbose.txt"),
    (["-vv", "--format", "color"], "verbose-color.txt"),
    (["-vv", "--format", "md"], "verbose-markdown.md"),
    (["-vv", "--format", "summary"], "verbose-summary.txt"),
])
def test_git_same(tmp_path, test_diff_renders, args, name):
    git_self_extract("0c197f2cdb0bf8c0ca95e76a837296fbebad436d", a := tmp_path / "a")
    git_self_extract("0c197f2cdb0bf8c0ca95e76a837296fbebad436d", b := tmp_path / "b")
    code, text = process2text([str(a), str(b), *args])
    sync_contents(cases / f"git/diff-same-{name}", text, test_diff_renders)
    assert code is False


@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["--format", "color"], "color.txt"),
    (["--format", "md"], "markdown.md"),
    (["--format", "summary"], "summary.txt"),
    (["--format", "color", "--text-line-split"], "line-split.txt"),
    (["--format", "html", "--text-line-split"], "line-split.html"),
])
def test_readme(test_diff_renders, args, name):
    code, text = process2text([str(cases / "readme/a.txt"), str(cases / "readme/b.txt"), *args])
    sync_contents(cases / f"readme/diff-{name}", text, test_diff_renders)
    assert code is True


@pytest.mark.parametrize("b", ["single-edit", "mosaic-edit", "rm-row", "add-row", "add-col", "rm-col", "not-a"])
@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["--format", "color"], "color.txt"),
    (["--format", "md"], "markdown.md"),
    (["--format", "summary"], "summary.txt"),
    (["--table-collapse"], "collapse.txt"),
    (["--format", "html"], "page.html"),
])
def test_co2_emissions(test_diff_renders, b, args, name):
    code, text = process2text([
        str(cases / "co2_emissions/a.csv"), str(cases / f"co2_emissions/b-{b}.csv"), "--min-ratio-row", "0.6", *args
    ])
    sync_contents(cases / f"co2_emissions/diff-{b}-{name}", text, test_diff_renders)
    assert code is True


@pytest.mark.parametrize("b", ["rename-col"])
@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["--align-col-data"], "precise.txt"),
    (["--align-col-data", "--format", "color"], "precise-color.txt"),
    (["--align-col-data", "--format", "md"], "precise-markdown.md"),
    (["--align-col-data", "--format", "summary"], "precise-summary.txt"),
    (["--align-col-data", "--table-collapse"], "collapse.txt"),
    (["--table-drop-cols", "Date"], "drop-cols.txt"),
])
def test_co2_emissions_2(test_diff_renders, b, args, name):
    code, text = process2text([
        str(cases / "co2_emissions/a.csv"), str(cases / f"co2_emissions/b-{b}.csv"), *args
    ])
    sync_contents(cases / f"co2_emissions/diff-{b}-{name}", text, test_diff_renders)
    assert code is True


@pytest.mark.parametrize("b", ["single-edit"])
@pytest.mark.parametrize("args, name", [
    (["--min-ratio-row", "1"], "min-ratio-row-1.txt"),
    (["--min-ratio-row", "0.79"], "min-ratio-row-079.txt"),
    (["--max-cost-row", "1"], "max-cost-row-1.txt"),
    (["--max-cost-row", "1", "--align-col-data"], "max-cost-row-1-acd.txt"),
    (["--max-cost-row", "2"], "max-cost-row-2.txt"),
    (["--min-ratio", "0.9", "--min-ratio-row", "1"], "tight-ratios.txt"),
    (["--max-cost", "1", "--min-ratio-row", "1"], "tight-ratios-2.txt"),
    (["--mime", "text/plain"], "mime.txt"),
    (["--context-size", "0"], "context-size-0.txt"),
    (["--table-sort"], "table-sort.txt"),
    (["--table-sort", "Kilotons of Co2"], "table-sort-1.txt"),
    (["--width", "5"], "short-5.txt"),
    (["--width", "15"], "short-15.txt"),
    (["--width", "25"], "short-25.txt"),
])
def test_co2_emissions_3(test_diff_renders, b, args, name):
    code, text = process2text([
        str(cases / "co2_emissions/a.csv"), str(cases / f"co2_emissions/b-{b}.csv"), "--min-ratio-row", "0.8", *args
    ])
    sync_contents(cases / f"co2_emissions/diff-{b}-{name}", text, test_diff_renders)
    assert code is True


@pytest.mark.parametrize("args, name", [
    ([], "default.txt"),
    (["--format", "color"], "color.txt"),
    (["--format", "md"], "markdown.md"),
    (["--format", "summary"], "summary.txt"),
    (["--table-collapse"], "collapse.txt"),
    (["--format", "html"], "page.html"),
])
def test_co2_emissions_4(test_diff_renders, args, name):
    code, text = process2text([
        str(cases / "co2_emissions/b-empty.csv"), str(cases / f"co2_emissions/b-empty-rm-col.csv"), "--mime", "text/csv", *args
    ])
    sync_contents(cases / f"co2_emissions/diff-empty-{name}", text, test_diff_renders)
    assert code is True


def test_pd_nans(tmp_path, test_diff_renders):
    code, text = process2text([
        str(cases / "pd_csv_nans/a.csv"), str(cases / "pd_csv_nans/b.csv"),
    ])
    sync_contents(cases / f"pd_csv_nans/diff.txt", text, test_diff_renders)
    assert code is True
