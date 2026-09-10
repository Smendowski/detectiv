from importlib.metadata import version

import detectiv


def test_version_matches_installed_metadata() -> None:
    assert detectiv.__version__ == version("detectiv")
