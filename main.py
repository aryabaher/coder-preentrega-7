"""Arranque de la API. La guardia corre antes de importar el grafo."""

from requisitos import exigir_python_312

exigir_python_312()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
