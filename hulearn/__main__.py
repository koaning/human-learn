import typer
import uvicorn

from hulearn import __version__
from hulearn.server import create_app

app = typer.Typer()


@app.command()
def version():
    typer.echo(__version__)


@app.command()
def serve(filepath: str, host: str = "0.0.0.0", port: int = 8000, label: str = None):
    app = create_app(filepath, label)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    app()
