import time
import asyncio
import pathlib

from pydantic import BaseModel, validator
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from clumper import Clumper

def create_app(filepath, label):
    """
    This function creates the app.
    """
    app = FastAPI()
    clump = Clumper.read_csv(filepath)
    dir = pathlib.Path(__file__).parent.absolute() / "static"

    @app.get("/", response_class=HTMLResponse)
    def index():
        path = pathlib.Path(dir) / "index.html"
        return path.read_text()

    @app.get("/info")
    def ping():
        return {"filepath": filepath, "label": label}

    @app.get("/dataset")
    async def dataset():
        return clump.collect()

    dir = pathlib.Path(__file__).parent.absolute() / "static"
    app.mount("/", StaticFiles(directory=dir), name="static")

    return app
