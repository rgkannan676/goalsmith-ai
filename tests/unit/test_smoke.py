import pytest

import goalsmith


def test_package_has_version() -> None:
    assert goalsmith.__version__ == "0.1.0"


def test_main_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    goalsmith.main()
    assert "goalsmith 0.1.0" in capsys.readouterr().out
