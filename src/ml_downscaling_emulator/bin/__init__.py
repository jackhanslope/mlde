import typer

from . import evaluate, postprocess

app = typer.Typer()
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(postprocess.app, name="postprocess")


if __name__ == "__main__":
    app()
