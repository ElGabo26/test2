
# SQL Agent RAG para Ollama vía OpenAI-compatible

## Arranque correcto

```bash
cd sql_agent_rag
pip install -r requirements.txt
python -m web.app
```

Abre:

```text
http://localhost:8000
```

## Verificación rápida

Prueba primero:

```text
http://localhost:8000/api/health
```

Si falla, revisa que Ollama esté levantado y que el modelo exista.

Variables opcionales:
- `OLLAMA_BASE_URL` por defecto: `http://localhost:11434/v1`
- `OLLAMA_MODEL` por defecto: `qwen2.5-coder:3b`
- `OLLAMA_API_KEY` por defecto: `ollama`
