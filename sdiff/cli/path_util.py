from pathlib import Path
import logging
from typing import Optional
from collections.abc import Iterator, Sequence, Callable
import re
import fnmatch
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchRule:
    accept: bool
    """
    A rule matching paths.
    
    Parameters
    ----------
    accept
        A flag indicating whether matching paths have to be accepted or declined.
    """

    def match(self, key: str) -> bool:
        """
        Attempts to match a path through the provided key.

        Parameters
        ----------
        key
            The key to match.

        Returns
        -------
        True if the key matches. This implementation always returns True.
        """
        return True

    def __str__(self):
        if self.accept:
            return "rule: include all"
        else:
            return "rule: exclude all"


accept_all = MatchRule(True)
reject_all = MatchRule(False)


@dataclass(frozen=True)
class RegexMatchRule(MatchRule):
    pattern: str
    pattern_str: str = None
    """
    A rule matching paths.
    
    Parameters
    ----------
    accept
        A flag indicating whether matching paths have to be accepted or declined.
    pattern
        Regex pattern to match.
    pattern_str
        The representation of the regex pattern.
    """

    def __post_init__(self):
        object.__setattr__(self, "pattern_str", self.pattern_str or self.pattern)

    @classmethod
    def from_glob(cls, accept: bool, glob_pattern: str):
        """
        Defines a regex pattern rule from the provided glob.

        Parameters
        ----------
        accept
            A flag indicating whether matching paths have to be accepted or declined.
        glob_pattern
            Glob pattern to match.

        Returns
        -------
        The match rule.
        """
        return RegexMatchRule(accept, fnmatch.translate(glob_pattern), glob_pattern)

    def match(self, key: str) -> bool:
        """
        Attempts to match a path through the provided key.

        Parameters
        ----------
        key
            The key to match.

        Returns
        -------
        True if the key matches.
        """
        return bool(re.fullmatch(self.pattern, key))

    def __str__(self):
        prefix = "include" if self.accept else "exclude"
        return f"{prefix} {self.pattern_str!r}"


glob_rule = RegexMatchRule.from_glob
accept_folders = glob_rule(True, "*/")


def iterdir(
    node: Path,
    root: Optional[Path] = None,
    rules: Sequence[MatchRule] = (accept_all,),
    sort: bool = False,
) -> Iterator[tuple[Path, MatchRule, str]]:
    """
    Recursive iteration over path tree using rsymc-like rules.

    Parameters
    ----------
    node
        The node to look into. Accepts folders or files or
        non-existent nodes but will yield only the existing and
        matching ones: whether they are files or folders.

        Symlinks are ignored.
    root
        The root which is used to construct keys for matching
        paths.
    rules
        The rules to use when iterating the path tree.
    sort
        If True, sorts files.

    Yields
    ------
    path
        A matching path.
    rule
        The rule mathing the path.
    key
        A key that matches the rule.
    """
    if not node.exists():
        return
    if node.is_symlink():
        logging.warning("ignoring symlink: %s", node)
        return
    if root is None:
        root = node
    key = str(node.relative_to(root))
    if node.is_dir():
        key += "/"
    for rule in rules:
        if rule.match(key):
            if rule.accept:
                yield node, rule, key
                if node.is_dir():
                    sub_nodes = node.iterdir()
                    if sort:
                        sub_nodes = sorted(sub_nodes)
                    for sub_node in sub_nodes:
                        yield from iterdir(sub_node, root, rules, sort=sort)
                break
            else:  # reject explicitly
                break


def iter_match(
    a: Path,
    b: Path,
    transform: Optional[Callable[[str], str]] = None,
    rules: Sequence[MatchRule] = (accept_all,),
    sort: bool = False,
    cherry_pick: Optional[str] = None,
) -> Iterator[tuple[Optional[Path], Optional[Path], str]]:
    """
    Iterates two path trees and matches the nodes.

    Parameters
    ----------
    a
    b
        The two paths to traverse.
    transform
        An optional transform for path keys.
    rules
        The rules to use when iterating the path trees.
    sort
        If True, sorts files.
    cherry_pick
        Once set, will only consider one file matching this argument.

    Yields
    ------
    path_a
    path_b
        Two matching paths. If one of the paths does not
        match the other one is set to None.
    key
        A key for both paths.
    """

    def _collect(_path: Path, _name: str) -> dict[str, Path]:
        _result = {}
        logging.info("collecting %s ...", _name)
        for _child, _rule, _key in iterdir(
            node=_path,
            root=_path,
            rules=rules,
            sort=sort,
        ):
            if _child.is_file():
                logging.debug("matched %s in %s: %s", _child, _name, _rule)
                if transform is not None:
                    _key = transform(_key)
                if _key in _result:
                    logging.warning(
                        "multiple paths transform into the same key %s", _key
                    )
                    logging.warning("  path: %s", _child)
                _result[_key] = _child
        return _result

    a_contents = _collect(a, "a")
    b_contents = _collect(b, "b")

    for key, a_path in a_contents.items():
        if cherry_pick is not None:
            try:
                next(re.finditer(cherry_pick, key))
            except StopIteration:
                continue
        try:
            b_path = b_contents.pop(key)
        except KeyError:
            yield a_path, None, key
        else:
            yield a_path, b_path, key
        if cherry_pick is not None:
            return

    for key, b_path in b_contents.items():
        if cherry_pick is not None:
            try:
                next(re.finditer(cherry_pick, key))
            except StopIteration:
                continue
        yield None, b_path, key
        if cherry_pick is not None:
            return
