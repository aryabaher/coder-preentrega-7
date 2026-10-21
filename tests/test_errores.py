import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import modelo
from app.graph import DecisionSupervisor, estado_inicial, obtener
from errors import (
    ClaveAPIError,
    ConfiguracionError,
    CuotaError,
    DecisionInvalidaError,
    OrquestadorError,
    RedError,
    SolicitudInvalidaError,
)
from tests.conftest import LLMFalso

PREGUNTA = "¿Cuántos días de vacaciones acumuló?"
_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_n = 0


def _ir(estado: dict) -> dict:
    global _n
    _n += 1
    return obtener().invoke(estado, {"configurable": {"thread_id": f"err-{_n}"}, "recursion_limit": 25})


def _status(codigo: int, clase: type[openai.APIStatusError], texto: str) -> openai.APIStatusError:
    return clase(texto, response=httpx.Response(codigo, request=_REQUEST), body=None)


def _cuota() -> openai.APIStatusError:
    return _status(429, openai.RateLimitError, "Rate limit reached")


class _Roto:
    def __init__(self, *fallos: BaseException, respuesta: DecisionSupervisor | None = None) -> None:
        self.fallos = list(fallos)
        self.llamadas = 0
        self.respuesta = respuesta or DecisionSupervisor(next="FINISH", razon="listo")

    def invoke(self, prompt: object, *args: object, **kwargs: object) -> DecisionSupervisor | AIMessage:
        self.llamadas += 1
        if self.fallos:
            raise self.fallos.pop(0)
        if isinstance(prompt, str):
            return AIMessage(content="En total, 231 días.")
        return self.respuesta

    def with_structured_output(self, schema: object, **kwargs: object) -> "_Roto":
        return self


def _usar(monkeypatch: pytest.MonkeyPatch, roto: _Roto) -> _Roto:
    monkeypatch.setattr(modelo, "obtener_llm", lambda: roto)
    return roto


def test_falta_la_clave_401(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ClaveAPIError, match=r"^401/key: falta OPENAI_API_KEY en las variables de entorno\."):
        modelo.crear_llm()


def test_401_no_se_reintenta(monkeypatch: pytest.MonkeyPatch) -> None:
    roto = _usar(monkeypatch, _Roto(_status(401, openai.AuthenticationError, "Incorrect API key provided")))
    with pytest.raises(ClaveAPIError, match=r"^401/key: el proveedor rechazó la API key\. No se reintenta"):
        _ir(estado_inicial(PREGUNTA))
    assert roto.llamadas == 1


def test_429_es_cuota_y_reintenta(monkeypatch: pytest.MonkeyPatch) -> None:
    roto = _usar(monkeypatch, _Roto(_cuota()))
    salida = _ir(estado_inicial(PREGUNTA))
    assert salida["task_completed"] is True
    assert roto.llamadas >= 2


def test_429_agotado_usa_el_ultimo_fallo(monkeypatch: pytest.MonkeyPatch) -> None:
    roto = _usar(monkeypatch, _Roto(_cuota(), _cuota(), _cuota()))
    with pytest.raises(CuotaError, match=r"^429/cuota: .*reintentos agotados: 3 intentos"):
        _ir(estado_inicial(PREGUNTA))
    assert roto.llamadas == 3


def test_timeout_es_red(monkeypatch: pytest.MonkeyPatch) -> None:
    _usar(monkeypatch, _Roto(TimeoutError("timed out"), TimeoutError("timed out"), TimeoutError("timed out")))
    with pytest.raises(RedError, match=r"^red/timeout: no se pudo hablar con el proveedor del LLM"):
        _ir(estado_inicial(PREGUNTA))


def test_backoff_duplica_la_espera(monkeypatch: pytest.MonkeyPatch) -> None:
    esperas: list[float] = []
    monkeypatch.setattr(modelo, "ESPERA_INICIAL_S", 0.5)
    monkeypatch.setattr("time.sleep", lambda segundos: esperas.append(segundos))
    _usar(monkeypatch, _Roto(_cuota(), _cuota(), _cuota()))
    with pytest.raises(CuotaError):
        _ir(estado_inicial(PREGUNTA))
    assert esperas == [0.5, 1.0]


def test_decision_ilegible_no_inventa_ruta(monkeypatch: pytest.MonkeyPatch) -> None:
    class Basura(_Roto):
        def invoke(self, *args: object, **kwargs: object) -> object:
            self.llamadas += 1
            return {"next": "nadie"}

    _usar(monkeypatch, Basura())
    with pytest.raises(DecisionInvalidaError, match=r"^Decisión inválida del supervisor:"):
        _ir(estado_inicial(PREGUNTA))


def test_pregunta_vacia_no_llega_al_grafo(dobles: tuple[LLMFalso, object, object]) -> None:
    llm, _, _ = dobles
    with pytest.raises(SolicitudInvalidaError, match=r"^Solicitud inválida: texto: String should have at least 1 character"):
        estado_inicial("   ")
    assert llm.supervisor.llamadas == 0


def test_provider_invalido(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "otro")
    with pytest.raises(ConfiguracionError, match=r"^LLM_PROVIDER inválido: 'otro'\. Valores soportados: google, openai, anthropic\.$"):
        modelo.crear_llm()


def test_error_no_transitorio_no_reintenta(monkeypatch: pytest.MonkeyPatch) -> None:
    roto = _usar(monkeypatch, _Roto(ValueError("algo raro")))
    with pytest.raises(OrquestadorError, match=r"^Error del LLM: ValueError: algo raro"):
        _ir(estado_inicial(PREGUNTA))
    assert roto.llamadas == 1


def test_crea_chat_openai_con_los_parametros(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "clave-de-test")
    llm = modelo.crear_llm("openai", modelo="gpt-4.1-mini", max_tokens=256)
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4.1-mini"


def test_crea_chat_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "clave-de-test")
    monkeypatch.setenv("LLM_PROVIDER", "google")
    llm = modelo.crear_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)


def test_crea_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-de-test")
    llm = modelo.crear_llm("anthropic")
    assert isinstance(llm, ChatAnthropic)
