import asyncio
import fnmatch
from collections.abc import AsyncIterator


class RedisFalso:
    """Lista y claves en memoria. blpop con timeout 0 espera, como Redis."""

    def __init__(self) -> None:
        self.datos: dict[str, str] = {}
        self.encolados: list[str] = []
        self.llamadas: list[tuple[str, str]] = []
        self._colas: dict[str, asyncio.Queue[str]] = {}

    def _cola(self, nombre: str) -> asyncio.Queue[str]:
        if nombre not in self._colas:
            self._colas[nombre] = asyncio.Queue()
        return self._colas[nombre]

    async def ping(self) -> bool:
        return True

    async def set(self, clave: str, valor: str, nx: bool = False) -> bool | None:
        if nx and clave in self.datos:
            return None
        self.llamadas.append(("set", valor))
        self.datos[clave] = valor
        return True

    async def get(self, clave: str) -> str | None:
        return self.datos.get(clave)

    async def delete(self, clave: str) -> int:
        return 1 if self.datos.pop(clave, None) is not None else 0

    async def scan_iter(self, match: str = "*") -> AsyncIterator[str]:
        for clave in list(self.datos):
            if fnmatch.fnmatch(clave, match):
                yield clave

    async def rpush(self, nombre: str, valor: str) -> int:
        self.llamadas.append(("rpush", valor))
        self.encolados.append(valor)
        await self._cola(nombre).put(valor)
        return self._cola(nombre).qsize()

    async def blpop(self, nombre: str, timeout: int = 0) -> tuple[str, str] | None:
        if timeout == 0:
            return nombre, await self._cola(nombre).get()
        try:
            valor = await asyncio.wait_for(self._cola(nombre).get(), timeout)
        except TimeoutError:
            return None
        return nombre, valor
