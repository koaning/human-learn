import typer

from hulearn import __version__


app = typer.Typer()


@app.command()
def version():
    typer.echo(__version__)


@app.command()
def hello():
    typer.echo("hello")


if __name__ == "__main__":
    app()
