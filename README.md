# API del orquestador, con Redis y aprobación humana

`POST /tasks` encola la consulta y devuelve un `job_id` al instante. El worker corre el orquestador del módulo 6 en un hilo, guarda el estado en Redis y, si la tarea pide una herramienta con efectos secundarios, pausa el grafo hasta `POST /tasks/{id}/approve`. Cada llamada al modelo sale como traza a un Phoenix local.

## Quick path

> **Requisito: Python 3.12 o superior.** `app/graph.py` declara `type Orquestador = CompiledStateGraph[...]`, el alias `type` de PEP 695, que no existe antes de 3.12. `main.py` y `app/main.py` llaman a `exigir_python_312()` antes de importar el grafo. Con 3.11 el mensaje sale antes del `SyntaxError`.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

En `.env` cargá la clave del proveedor (`LLM_PROVIDER=openai` y `OPENAI_API_KEY` para que Phoenix calcule el costo), `REDIS_URL=redis://localhost:6379` y, si ya tenés el índice, `PINECONE_API_KEY`. Si `PINECONE_API_KEY` queda vacía, la búsqueda corre en memoria. `.env` no se versiona.

Redis 8, porque `RedisSaver.setup()` crea índices de búsqueda:

```text
docker run -d --name redis-preentrega-7 -p 6379:6379 redis:8
```

Phoenix, en otra terminal y con el mismo entorno:

```text
python -m phoenix.server.main serve
```

El dashboard queda en http://localhost:6006. El colector es `http://localhost:6006/v1/traces`, proyecto `preentrega-7`.

La API:

```text
python main.py
```

Chequeo sin clave y sin Redis, y los tests:

```text
python validacion.py
python -m pytest -q
py -3.11 main.py
```

`python validacion.py` y `pytest` no llaman a la API: el doble del modelo está en `tests/conftest.py` y, para la demo, en `validacion.py`.

Con Docker Compose (Redis 8, Phoenix y la API):

```text
docker compose up --build
```

## Probar una tarea

**Windows (PowerShell):**

```powershell
curl.exe -X POST localhost:8000/tasks -H "Content-Type: application/json" -d "{\"query\": \"¿Cuántos días de vacaciones acumuló un empleado con 12 años?\"}"
curl.exe localhost:8000/tasks/<job_id>
```

**Linux/macOS:**

```bash
curl -X POST localhost:8000/tasks -H "Content-Type: application/json" -d '{"query": "¿Cuántos días de vacaciones acumuló un empleado con 12 años?"}'
curl localhost:8000/tasks/<job_id>
```

El POST vuelve enseguida con `job_id` y `"status": "PENDING"`. El GET no corre el agente: lee Redis. El ciclo es `PENDING`, `RUNNING` y `DONE`. Si el worker lanza, el estado queda `FAILED` y el cliente deja de esperar.

## Cinco peticiones concurrentes

Con la API, Redis y Phoenix ya levantados, y el proyecto `preentrega-7` vacío (el script lo borra antes de enviar):

```text
python carga.py
```

Manda 5 `POST /tasks` a la vez, comprueba que vuelven en menos de un segundo con cinco `job_id` distintos en `PENDING`, y espera a que las cinco queden `DONE`. El costo por ejecución y la latencia p95 se leen en el dashboard, no en la consola: Phoenix los calcula con los tokens de entrada y salida. Las capturas están en `screenshots/`.

En la corrida que quedó capturada, las cinco trazas se llaman `orquestador`. Phoenix les calculó estas latencias: 8.23 s, 8.88 s, 9.31 s, 9.61 s y 11.04 s. El p95 de ese conjunto es 10.75 s. El costo de cada una, también calculado por Phoenix, fue 0.00089, 0.00092, 0.00103, 0.00116 y 0.00150 USD. En Metrics el gráfico de costo de la corrida marca `$6.00e-3` y la serie P95 del gráfico de latencia cae en esa misma franja de 9 a 11 s.

## Aprobación humana

Una consulta normal (políticas y un cálculo) llega a `DONE` sin pausa. Si el texto pide registrar un ajuste de nómina, el nodo `aprobacion` llama a `interrupt` antes de `registrar_ajuste_nomina`. El GET pasa a `AWAITING_APPROVAL` y trae `interrupt.tool`.

```powershell
curl.exe -X POST localhost:8000/tasks/<job_id>/approve -H "Content-Type: application/json" -d "{\"approved\": true}"
```

`approved: true` reanuda y la herramienta asienta el ajuste. `approved: false` reanuda y cierra sin ejecutarla. El POST de approve también vuelve al instante, con `"status": "RUNNING"`.

## Archivos

| Artefacto | Dónde está |
| --- | --- |
| `POST /tasks`, `GET /tasks/{id}`, `POST /tasks/{id}/approve` | `app/main.py` |
| `TaskRequest`, `TaskResponse`, `TaskStatus` | `app/main.py` |
| Orquestador, `RedisSaver`, `thread_id` | `app/graph.py` |
| Worker `PENDING` / `RUNNING` / `FAILED` / `DONE` | `app/worker.py` |
| OpenInference, `LangChainInstrumentor`, decorador `trazar_llm` | `app/observability.py` |
| `interrupt` y `registrar_ajuste_nomina` | `app/hitl.py` |
| Investigador, analista, RAG | `agents/`, `rag.py`, `data/` |
| Familias de error del modelo | `errors.py`, `modelo.py` |
| Cinco POST concurrentes | `carga.py` |
| Demo sin clave | `validacion.py` |
| Tests | `tests/` |
| Redis 8 + Phoenix + API | `docker-compose.yml` |
| Capturas del dashboard | `screenshots/` |

## Estados

| Estado | Cuándo |
| --- | --- |
| `PENDING` | El POST ya escribió Redis y encoló. Todavía no corre el grafo. |
| `RUNNING` | El worker tomó el job, o acaba de entrar un approve. |
| `AWAITING_APPROVAL` | El grafo está pausado en `interrupt`. Hace falta `POST /tasks/{id}/approve`. |
| `DONE` | Terminó. `result` es la respuesta. |
| `FAILED` | El worker capturó la excepción. `error` dice qué pasó. |

El `set` de `PENDING` ocurre antes del `rpush`. El grafo corre con `asyncio.to_thread`, así el event loop sigue respondiendo el GET. `recursion_limit` (default 25) llega al `invoke`.

Los checkpoints del hilo viven en `RedisSaver`, con `thread_id` igual al `job_id`. El estado que consulta el cliente es la clave `status:{job_id}`.

## Errores

### 401/key

Mensaje, si falta la variable:

```text
401/key: falta OPENAI_API_KEY en las variables de entorno. Copiá .env.example a .env y cargá la clave. No se reintenta.
```

Si la clave está y el proveedor la rechaza:

```text
401/key: el proveedor rechazó la API key. No se reintenta: revisá la clave en .env.
```

No se reintenta. Si ocurre dentro del worker, el job queda `FAILED` con ese texto.

Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_falta_la_clave_401 tests/test_errores.py::test_401_no_se_reintenta -q`. El job en segundo plano lo cubre `tests/test_api.py::test_401_en_segundo_plano_queda_failed`.

### 429/cuota

Mensaje:

```text
429/cuota: el proveedor limitó las llamadas (rate limit o cuota). Detalle: RateLimitError: Rate limit reached
```

`invocar` reintenta con backoff (0.5 s, 1 s). Si se agotan los 3 intentos, el mensaje es el del último fallo más ` (reintentos agotados: 3 intentos; probá de nuevo en unos segundos)`. No es un texto fijo.

Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_429_es_cuota_y_reintenta tests/test_errores.py::test_429_agotado_usa_el_ultimo_fallo -q`.

### red/timeout

Mensaje:

```text
red/timeout: no se pudo hablar con el proveedor del LLM. Detalle: TimeoutError: timed out
```

También se reintenta. Una 401 no entra en esta familia.

Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_timeout_es_red -q`.

### Output truncado

Mensaje, como aporte del investigador:

```text
ERROR: output truncado: investigador cortó la respuesta por max_tokens. Pedile una respuesta más corta.
```

`es_truncada` mira `finish_reason == "length"`. Ese aporte no cuenta como tarea cumplida. El job no pasa a `FAILED` por esto: el supervisor sigue.

Cómo reproducirlo: `python -m pytest tests/test_grafo.py::test_respuesta_truncada_no_se_toma_como_dato -q`.

### Solicitud inválida

En el grafo, antes de gastar una llamada:

```text
Solicitud inválida: texto: String should have at least 1 character
```

Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_pregunta_vacia_no_llega_al_grafo -q`.

En la API, la misma query en blanco responde 422:

```text
{"detail":[{"type":"string_too_short","loc":["body","query"],"msg":"String should have at least 1 character","input":"   ","ctx":{"min_length":1}}]}
```

Un body `{}` también es 422, con `loc` en `["body", "query"]`. Cómo reproducirlo: `python -m pytest tests/test_api.py::test_http_job_inexistente_y_body_invalido -q`.

### Job que no existe

```text
{"detail":"Job not found"}
```

`GET /tasks/{job_id}` y `POST /tasks/{job_id}/approve` hacen `raise HTTPException(status_code=404, detail="Job not found")`. Cómo reproducirlo: `curl.exe localhost:8000/tasks/inventado`. Lo cubren `test_job_inexistente_responde_404` y `test_http_job_inexistente_y_body_invalido`.

### Approve sin pausa

```text
{"detail":"Job is not awaiting approval"}
```

El job existe pero no está en `AWAITING_APPROVAL` (sigue `PENDING`, `RUNNING` o ya está `DONE`). HTTP 409. Cómo reproducirlo: `python -m pytest tests/test_api.py::test_approve_sin_pausa_responde_409 -q`.

### El worker falla

Estado guardado en Redis:

```text
{"status": "FAILED", "query": "una", "result": null, "error": "corte de prueba"}
```

La excepción queda en `error` y el bucle sigue con el job siguiente. Cómo reproducirlo: `python -m pytest tests/test_api.py::test_el_worker_guarda_failed_y_sigue -q`.

### Redis apagado

El POST no llega a encolar. La respuesta es 503:

```text
{"detail":"Redis is unavailable"}
```

La librería, si se la deja salir, dice `Error 22 connecting to localhost:6379. El equipo remoto rechazó la conexión de red.` Cómo reproducirlo: no levantar Redis y hacer `POST /tasks`, o `python -m pytest tests/test_api.py::test_redis_apagado_no_encola -q`.

### Decisión ilegible

```text
Decisión inválida del supervisor:
```

Si la salida estructurada no es `DecisionSupervisor`, no se inventa una ruta. Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_decision_ilegible_no_inventa_ruta -q`.

### LLM_PROVIDER inválido

```text
LLM_PROVIDER inválido: 'otro'. Valores soportados: google, openai, anthropic.
```

Cómo reproducirlo: `python -m pytest tests/test_errores.py::test_provider_invalido -q`.

## Evidencias

| Archivo | Qué muestra |
| --- | --- |
| `evidencias/01-validacion.txt` | Guardia 3.11, 401/key, 429, red/timeout, pausa HITL, ciclo PENDING → AWAITING_APPROVAL → DONE, 404, 409, 422, 503 y FAILED. `git ls-files` no lista `.env`. |
| `evidencias/02-pytest.txt` | 51 passed. |
| `evidencias/03-python-311.txt` | `py -3.11 main.py` corta con el mensaje de la guardia. |
| `evidencias/04-carga.txt` | Los 5 POST concurrentes, todos en `DONE`. |
| `screenshots/01-trazas.png` | Una traza `orquestador`: supervisor, investigador, analista y calculadora. |
| `screenshots/02-costo-por-ejecucion.png` | Vista Metrics. El gráfico Cost marca `$6.00e-3` y el desglose de `gpt-4o-mini`. |
| `screenshots/03-latencia-p95.png` | La misma vista Metrics. El gráfico Trace latency incluye la serie P95, en la franja de 9 a 11 s. |

## Checklist

- [x] `POST /tasks` encola y devuelve `TaskResponse(job_id, status="PENDING")` sin esperar al grafo
- [x] `GET /tasks/{job_id}` responde `TaskStatus` y, si no existe, `detail="Job not found"`
- [x] El estado vive en Redis: `PENDING`, `RUNNING`, `FAILED`, `DONE`. Una excepción del worker deja `FAILED`
- [x] El orquestador compila con `RedisSaver` y `thread_id`
- [x] OpenInference exporta cada llamada (`trazar_llm`, `LangChainInstrumentor`) a Phoenix
- [x] El nodo `aprobacion` usa `interrupt` y `POST /tasks/{id}/approve` reanuda
- [x] `python carga.py` manda las 5 peticiones concurrentes
- [x] `requirements.txt`, Docker Compose, README de Redis + API + la corrida de 5
- [x] Capturas del dashboard en `screenshots/`: trazas, costo por ejecución y latencia p95
