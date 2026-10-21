import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as api
from app import worker
from app.worker import STATUS_PREFIX, RedisUnavailable, recover_orphans
from tests.redis_falso import RedisFalso

RAIZ = Path(__file__).resolve().parents[1]


async def test_running_huerfano_pasa_a_failed(redis: RedisFalso) -> None:
    for job_id, estado in (("corriendo", "RUNNING"), ("en-pausa", "AWAITING_APPROVAL"), ("listo", "DONE")):
        await redis.set(
            f"{STATUS_PREFIX}{job_id}",
            json.dumps({"job_id": job_id, "thread_id": job_id, "status": estado, "query": "q", "result": None}),
        )
        await redis.set(f"thread:{job_id}", job_id)
    recuperados = await recover_orphans()
    assert recuperados == ["corriendo"]
    corriendo = json.loads(await redis.get(f"{STATUS_PREFIX}corriendo"))
    assert corriendo["status"] == "FAILED"
    assert corriendo["error"] == "worker restarted before the job finished"
    assert await redis.get("thread:corriendo") is None
    assert json.loads(await redis.get(f"{STATUS_PREFIX}en-pausa"))["status"] == "AWAITING_APPROVAL"
    assert await redis.get("thread:en-pausa") == "en-pausa"


async def test_redis_caido_al_arrancar_dice_que_hacer(monkeypatch: pytest.MonkeyPatch) -> None:
    class SinRedis:
        async def ping(self) -> bool:
            raise ConnectionError("Error 10061 connecting to localhost:6379.")

    monkeypatch.setattr(worker, "redis_client", None)
    monkeypatch.setattr(worker.aioredis, "from_url", lambda *a, **k: SinRedis())
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    with pytest.raises(RedisUnavailable) as exc:
        await worker.connect()
    assert str(exc.value) == (
        "Redis is unavailable at redis://localhost:6379. "
        "Start it with: docker run -d --name redis-preentrega-7 -p 6379:6379 redis:8"
    )


def test_la_api_sin_worker_no_consume_la_cola(redis: RedisFalso, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKER_IN_API", "0")
    llamadas: list[str] = []

    async def no_debe_correr() -> None:
        llamadas.append("worker")

    monkeypatch.setattr(worker, "task_worker", no_debe_correr)
    monkeypatch.setattr(worker, "recover_orphans", no_debe_correr)
    with TestClient(api.app) as cliente:
        respuesta = cliente.post("/tasks", json={"query": "hola"})
    assert respuesta.status_code == 200
    assert llamadas == []
    assert len(redis.encolados) == 1


def test_el_worker_tiene_entrypoint_propio() -> None:
    fuente = (RAIZ / "app" / "worker.py").read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in fuente
    assert "await recover_orphans()" in fuente
    compose = (RAIZ / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'command: ["python", "-m", "app.worker"]' in compose
    assert 'WORKER_IN_API: "0"' in compose
    salida = subprocess.run(
        [sys.executable, "-c", "import app.worker as w; print(w.main.__name__)"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
        env={**__import__("os").environ, "TRAZAS_OFFLINE": "1"},
    )
    assert salida.returncode == 0, salida.stderr
    assert salida.stdout.strip() == "main"
