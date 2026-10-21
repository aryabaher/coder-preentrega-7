import ast
import sys
from pathlib import Path

import pytest

from requisitos import exigir_python_312

RAIZ = Path(__file__).resolve().parents[1]


def test_python_312_o_mayor_pasa() -> None:
    exigir_python_312((3, 12, 0))
    exigir_python_312()
    assert sys.version_info >= (3, 12)


def test_python_311_se_rechaza_con_el_mensaje() -> None:
    with pytest.raises(SystemExit) as error:
        exigir_python_312((3, 11, 9))
    assert "Python 3.12+ requerido" in str(error.value)
    assert "3.11" in str(error.value)


def test_la_guardia_corre_antes_de_importar_el_grafo() -> None:
    for relativo in ("main.py", "app/main.py"):
        lineas = (RAIZ / relativo).read_text(encoding="utf-8").splitlines()
        guardia = next(i for i, linea in enumerate(lineas) if linea.strip() == "exigir_python_312()")
        posteriores = [
            i
            for i, linea in enumerate(lineas)
            if "app.graph" in linea or "app.worker" in linea or "uvicorn" in linea
        ]
        assert posteriores and guardia < min(posteriores)


def test_el_grafo_no_parsea_en_311_y_el_arranque_si() -> None:
    ast.parse((RAIZ / "requisitos.py").read_text(encoding="utf-8"), feature_version=(3, 11))
    ast.parse((RAIZ / "main.py").read_text(encoding="utf-8"), feature_version=(3, 11))
    ast.parse((RAIZ / "app" / "main.py").read_text(encoding="utf-8"), feature_version=(3, 11))
    with pytest.raises(SyntaxError):
        ast.parse((RAIZ / "app" / "graph.py").read_text(encoding="utf-8"), feature_version=(3, 11))
