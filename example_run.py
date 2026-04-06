from src.sql_rag_pipeline import build_client, generate_sql, validate_sql

client = build_client()
r = generate_sql(
    'ventas de la marca TONI por mes en 2025',
    model='qwen2.5-coder:3b',
    detector_model='qwen2.5-coder:3b',
    client=client,
)
print(r['prompt_chars'])
print(r['detected_entities'])
print(r['prompt'])
print(r['sql'])
print(validate_sql(r['sql'], r['validation_context']))
