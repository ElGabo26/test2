# SQL Agent RAG para Ollama vía OpenAI-compatible

Incluye catálogo, joins, reglas, ejemplos y pipeline con prompt <=1500 caracteres.


## Interfaz Flask

Ejecuta la interfaz web así:

```bash
python web/app.py
```

Luego abre:

```text
http://localhost:8000
```

La UI permite:
- ingresar una pregunta
- escoger el modelo de Ollama
- ver el prompt final y su longitud
- revisar la validación
- intentar corrección automática cuando falle la primera salida
