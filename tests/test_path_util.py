import pytest

from sdiff.cli.path_util import (
    iterdir,
    iter_match,
    accept_all,
    accept_folders,
    reject_all,
    glob_rule,
)


def test_no_files(tmp_path):
    assert set(iterdir(tmp_path, rules=[reject_all])) == set()


def test_no_files_r(tmp_path):
    assert set(iterdir(tmp_path, rules=[accept_all])) == {
        (tmp_path, accept_all, "./"),
    }


def test_simple(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").touch()

    assert set(iterdir(tmp_path, rules=[reject_all])) == set()


def test_simple_r(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").touch()

    assert set(iterdir(tmp_path, rules=[accept_all])) == {
        (tmp_path, accept_all, "./"),
        (tmp_path / "1.txt", accept_all, "1.txt"),
        (tmp_path / "data", accept_all, "data"),
    }


def test_simple_depth_r(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "2.txt").touch()

    assert set(iterdir(tmp_path, rules=[accept_all])) == {
        (tmp_path, accept_all, "./"),
        (tmp_path / "1.txt", accept_all, "1.txt"),
        (tmp_path / "data", accept_all, "data/"),
        (tmp_path / "data/2.txt", accept_all, "data/2.txt"),
    }


def test_simple_exclude(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").touch()

    rules = [glob_rule(False, "1.txt"), accept_all]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, accept_all, "./"),
        (tmp_path / "data", accept_all, "data"),
    }


def test_simple_include(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").touch()

    rules = [glob_rule(True, "1.txt"), accept_folders, reject_all]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, accept_folders, "./"),
        (tmp_path / "1.txt", rules[0], "1.txt"),
    }


def test_nested(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "1.txt").touch()

    rules = [
        glob_rule(True, "1.txt"),
        glob_rule(True, "data/"),
        glob_rule(True, "./"),
    ]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, rules[2], "./"),
        (tmp_path / "1.txt", rules[0], "1.txt"),
        (tmp_path / "data", rules[1], "data/"),
    }


def test_nested_all(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "1.txt").touch()

    rules = [
        glob_rule(True, "*1.txt"),
        accept_folders,
    ]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, accept_folders, "./"),
        (tmp_path / "1.txt", rules[0], "1.txt"),
        (tmp_path / "data", accept_folders, "data/"),
        (tmp_path / "data/1.txt", rules[0], "data/1.txt"),
    }


def test_nested_some(tmp_path):
    (tmp_path / "1.txt").touch()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "1.txt").touch()

    rules = [
        glob_rule(True, "*1.txt"),
        glob_rule(True, "./"),
    ]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, rules[1], "./"),
        (tmp_path / "1.txt", rules[0], "1.txt"),
    }


def test_nested_complex(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/important").mkdir()
    (tmp_path / "data/important/file.txt").touch()
    (tmp_path / "data/unimportant.txt").touch()
    (tmp_path / "data/stuff").mkdir()
    (tmp_path / "data/stuff/something.txt").touch()

    rules = [
        glob_rule(True, "./"),
        glob_rule(True, "data/"),
        glob_rule(True, "data/important/"),
        glob_rule(True, "data/important/*"),
        glob_rule(True, "data/stuff/something.txt"),
    ]

    assert set(iterdir(tmp_path, rules=rules)) == {
        (tmp_path, rules[0], "./"),
        (tmp_path / "data", rules[1], "data/"),
        (tmp_path / "data/important", rules[2], "data/important/"),
        (tmp_path / "data/important/file.txt", rules[3], "data/important/file.txt"),
    }


@pytest.fixture
def a(tmp_path):
    result = tmp_path / "a"
    result.mkdir()
    return result


@pytest.fixture
def b(tmp_path):
    result = tmp_path / "b"
    result.mkdir()
    return result


def test_match_no_files(a, b):
    assert set(iter_match(a, b)) == set()


def test_match_one_file(a, b):
    (a / "file.txt").touch()
    (b / "file.txt").touch()

    assert set(iter_match(a, b)) == {
        (a / "file.txt", b / "file.txt", "file.txt"),
    }
