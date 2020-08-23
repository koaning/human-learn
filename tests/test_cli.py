from typer.testing import CliRunner

from hulearn.__main__ import app
from hulearn import __version__

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
