
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.sql_rag_pipeline import generate_sql, repair_sql, validate_sql, build_client

app = Flask(__name__, template_folder="templates", static_folder="static")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    low = msg.lower()
    if "connection" in low or "refused" in low or "failed to establish" in low:
        return (
            "No se pudo conectar con Ollama. Verifica que el servicio esté levantado y que "
            f"OLLAMA_BASE_URL apunte correctamente a {OLLAMA_BASE_URL}."
        )
    if "model" in low and ("not found" in low or "404" in low):
        return "El modelo indicado no existe en Ollama o no está descargado."
    if "api key" in low:
        return "Error de configuración del cliente OpenAI-compatible."
    return msg or exc.__class__.__name__


@app.get("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL)


@app.get("/api/health")
def api_health():
    try:
        client = build_client(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        # Llamada ligera para verificar conectividad con Ollama
        models = client.models.list()
        names = [getattr(m, 'id', None) or getattr(m, 'model', None) for m in getattr(models, 'data', [])]
        return jsonify({
            "ok": True,
            "ollama_base_url": OLLAMA_BASE_URL,
            "models": [n for n in names if n][:20],
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "ollama_base_url": OLLAMA_BASE_URL,
            "error": _friendly_error(exc),
            "raw_error": str(exc),
        }), 500


@app.post("/api/generate")
def api_generate():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    model = str(payload.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    autorepair = bool(payload.get("autorepair", True))

    if not question:
        return jsonify({"ok": False, "error": "La pregunta es obligatoria."}), 400

    try:
        client = build_client(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        result = generate_sql(question=question, model=model, client=client)
        validation = validate_sql(result["sql"])

        repaired_sql = None
        repaired_validation = None

        if autorepair and not validation["valid"] and result["sql"] != "NO_SQL":
            repaired_sql = repair_sql(
                question=question,
                bad_sql=result["sql"],
                validation_errors=validation["errors"],
                model=model,
                client=client,
            )
            repaired_validation = validate_sql(repaired_sql)

        return jsonify(
            {
                "ok": True,
                "question": question,
                "model": model,
                "prompt": result["prompt"],
                "prompt_chars": result["prompt_chars"],
                "intent": result["intent"],
                "context": result["context"],
                "sql": result["sql"],
                "validation": validation,
                "repaired_sql": repaired_sql,
                "repaired_validation": repaired_validation,
            }
        )
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": _friendly_error(exc),
            "raw_error": str(exc),
            "question": question,
            "model": model,
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
