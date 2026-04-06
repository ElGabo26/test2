# SQL Agent RAG para Ollama vía OpenAI-compatible

## Cambios incluidos en test6

- contexto máximo de 3000 caracteres
- capa previa de detección de entidades categóricas con modelo
- la búsqueda semejante de strings solo se habilita cuando el detector encuentra una entidad explícita
- los valores detectados se fuerzan a MAYÚSCULAS para `LIKE`
- siempre se hace `JOIN` con `DIM_FECHA` y el filtro de fechas usa `DF.FECHA`
- cuando el modelo devuelve `NO_SQL`, la API retorna `no_sql_reason`

## Arranque

```bash
cd sql_agent_rag
pip install -r requirements.txt
python -m web.app
```

Abre:

```text
http://localhost:8000
```

## Variables opcionales

- `OLLAMA_BASE_URL` por defecto: `http://localhost:11434/v1`
- `OLLAMA_MODEL` por defecto: `qwen2.5-coder:3b`
- `OLLAMA_DETECTOR_MODEL` por defecto: igual al modelo generador
- `OLLAMA_API_KEY` por defecto: `ollama`

## Lógica nueva de entidades

1. El pipeline primero detecta si el usuario habló explícitamente de una `marca`, `familia`, `empresa`, `unidad de negocio` u otro valor categórico soportado.
2. Solo si esa detección es positiva, el prompt autoriza búsquedas del tipo `UPPER(columna) LIKE '%VALOR%'`.
3. Si no se detecta entidad explícita, el modelo recibe la orden de **no** usar filtros de texto semejantes.
4. La detección se retroalimenta al generador indicando tipo de entidad, valor normalizado y columnas candidatas `NOM_`/`DESC_`.

## Carpeta deploy

Además del proyecto completo, se genera `deploy.zip` con el pipeline listo para integrarlo en otro proyecto sin la interfaz Flask.
