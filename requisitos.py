"""Chequeo de versión de Python. Se importa antes que el resto: tiene que parsear en versiones viejas."""

from __future__ import annotations

import sys
from collections.abc import Sequence

VERSION_MINIMA = (3, 12)


def exigir_python_312(version: Sequence[int] | None = None) -> None:
    """Corta con un mensaje claro si el intérprete es anterior a 3.12.

    app/graph.py usa el alias `type` (PEP 695), que en 3.11 o antes es un
    SyntaxError al importar. Este chequeo corre antes de ese import.
    """
    actual = tuple(version or sys.version_info)[:2]
    if actual < VERSION_MINIMA:
        raise SystemExit(
            "Python 3.12+ requerido: app/graph.py usa el alias `type` (PEP 695). "
            f"Versión actual: {actual[0]}.{actual[1]}. Creá el venv con `py -3.12 -m venv .venv` "
            "(Windows) o `python3.12 -m venv .venv` (Linux/macOS)."
        )
