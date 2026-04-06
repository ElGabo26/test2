from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from sql_rag_pipeline import generate_sql, repair_sql, validate_sql

app = Flask(__name__, template_folder="templates", static_folder="static")

DEFAULT_MODEL = "qwen2.5-coder:3b"


@app.get("/")
def index():
    return render_template("index.html", default_model=DEFAULT_MODEL)


@app.post("/api/generate")
def api_generate():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    model = str(payload.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    autorepair = bool(payload.get("autorepair", True))

    if not question:
        return jsonify({"ok": False, "error": "La pregunta es obligatoria."}), 400

    try:
        result = generate_sql(question=question, model=model)
        validation = validate_sql(result["sql"])

        repaired_sql = None
        repaired_validation = None

        if autorepair and not validation["valid"] and result["sql"] != "NO_SQL":
            repaired_sql = repair_sql(
                question=question,
                bad_sql=result["sql"],
                validation_errors=validation["errors"],
                model=model,
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
    except Exception as exc:  # pragma: no cover
        return jsonify({"ok": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
