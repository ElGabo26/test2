# SQL Agent RAG para Ollama vía OpenAI-compatible

## Cambios incluidos en esta versión

- contexto máximo de 3000 caracteres
- bloque visual para tablas utilizadas en la interfaz Flask
- priorización de columnas `NOM_*` y `DESC_*` para búsqueda y respuesta
- rango temporal por defecto desde `2025-01-01` hasta la fecha actual
- uso obligatorio de `DIM_FECHA` para filtrar fechas

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

Variables opcionales:
- `OLLAMA_BASE_URL` por defecto: `http://localhost:11434/v1`
- `OLLAMA_MODEL` por defecto: `qwen2.5-coder:3b`
- `OLLAMA_API_KEY` por defecto: `ollama`


## Cambios recientes
- Contexto máximo: 3000 caracteres.
- Los verbos de instrucción como `GENERA`, `MUESTRA` o `DAME` ya no se convierten en términos de búsqueda.
- Cuando la salida es `NO_SQL`, la API devuelve una explicación corta en `no_sql_reason`.
