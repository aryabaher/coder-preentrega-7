import asyncio


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

    async def set(self, clave: str, valor: str) -> bool:
        self.llamadas.append(("set", valor))
        self.datos[clave] = valor
        return True

    async def get(self, clave: str) -> str | None:
        return self.datos.get(clave)

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
