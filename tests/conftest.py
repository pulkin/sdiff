import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--test-diff-renders",
        action="store_true",
        help="tests whether diff renders are the same as those in the source tree",
    )


@pytest.fixture(scope="session")
def test_diff_renders(pytestconfig):
    """Description of changes triggered by parameter."""
    return pytestconfig.getoption("--test-diff-renders")
