"""Errores controlados del orquestador. Cada familia tiene su prefijo en el mensaje."""


class OrquestadorError(Exception):
    """Fallo del orquestador convertido en error de aplicación."""

    reintentable: bool = False


class ClaveAPIError(OrquestadorError):
    """401/key: falta la API key o el proveedor la rechazó. No se reintenta."""


class CuotaError(OrquestadorError):
    """429/cuota: rate limit o cuota del proveedor. Se reintenta con backoff."""

    reintentable = True


class RedError(OrquestadorError):
    """red/timeout: corte de red, timeout o 5xx del proveedor. Se reintenta con backoff."""

    reintentable = True


class SalidaTruncadaError(OrquestadorError):
    """El supervisor cortó su decisión por max_tokens. Sin decisión completa no hay ruta."""


class DecisionInvalidaError(OrquestadorError):
    """La salida estructurada del supervisor no se pudo leer como Decision."""


class SolicitudInvalidaError(OrquestadorError):
    """La solicitud no pasa la validación Pydantic (texto vacío, límites fuera de rango)."""


class LimiteRecursionError(OrquestadorError):
    """El grafo llegó a recursion_limit. Con las guardas del supervisor no debería pasar."""


class ConfiguracionError(OrquestadorError):
    """LLM_PROVIDER u otra variable de entorno con un valor no soportado."""


class IndiceIncompatibleError(OrquestadorError):
    """El índice existente no coincide en dimensión o métrica con el embedding. No se reingesta."""
