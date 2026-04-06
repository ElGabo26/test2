from src.sql_rag_pipeline import generate_sql, validate_sql

r=generate_sql('ventas por mes y categoria contable en 2025', model='qwen2.5-coder:3b')
print(r['prompt_chars'])
print(r['prompt'])
print(r['sql'])
print(validate_sql(r['sql']))
