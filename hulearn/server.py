import pathlib

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import pandas as pd


def create_app(filepath, label):
    """
    This function creates the app.
    """
    app = FastAPI()
    df = pd.read_csv(filepath)
    dir = pathlib.Path(__file__).parent.absolute() / "static"

    @app.get("/", response_class=HTMLResponse)
    def index():
        path = pathlib.Path(dir) / "index.html"
        return path.read_text()

    @app.get("/info")
    def ping():
        return {
            "filepath": filepath,
            "label": label,
            "columns": {
                k: "num" if v == "O" else "str" for k, v in dict(df.dtypes).items()
            },
        }

    @app.get("/dataset")
    async def dataset():
        return df.to_dict(orient="records")

    @app.get("/uniq_labels")
    async def uniq_labels():
        if label not in df.columns:
            return []
        return list(df[label].unique())

    dir = pathlib.Path(__file__).parent.absolute() / "static"
    app.mount("/", StaticFiles(directory=dir), name="static")

    return app
