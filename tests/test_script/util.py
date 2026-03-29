import tarfile
from io import StringIO
from multiprocessing import set_start_method
from operator import eq
from subprocess import Popen, check_output
from tempfile import NamedTemporaryFile

from sdiff.cli.processor import process_print, run

set_start_method("forkserver")


def diff2text(a, b, **kwargs):
    buffer = StringIO()
    process_print(a, b, output_file=buffer, sort=True, **kwargs)
    return buffer.getvalue()


def process2text(args, sort=True):
    if sort:
        args = [*args, "--sort"]
    with NamedTemporaryFile("w+") as f:
        exit_code = run(["--output", f.name, "--width", "160", *args])
        f.seek(0)
        return exit_code, f.read()


def git_self_extract(commit, dst):
    root = check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    with NamedTemporaryFile("w+b") as f:
        Popen(["git", "-C", root, "archive", commit], stdout=f).communicate()
        f.flush()
        f.seek(0)
        with tarfile.open(fileobj=f) as f_tar:
            f_tar.extractall(dst, filter="data")


def sync_contents(path, content, check, eq=eq):
    with open(path, "r" if check else "w") as f:
        if check:
            assert eq(content, f.read())
        else:
            f.write(content)
