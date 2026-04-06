
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BASE_DIR / "catalog"
EXAMPLES_DIR = BASE_DIR / "examples"


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SCHEMA_CATALOG = load_json(CATALOG_DIR / "schema_catalog.json")
JOIN_WHITELIST = load_json(CATALOG_DIR / "join_whitelist.json")
BUSINESS_RULES = load_json(CATALOG_DIR / "business_rules.json")
BUSINESS_GLOSSARY = load_json(CATALOG_DIR / "business_glossary.json")


def load_examples() -> List[Dict[str, Any]]:
    with open(EXAMPLES_DIR / "sql_examples.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


SQL_EXAMPLES = load_examples()


@dataclass
class NormalizedIntent:
    question: str
    domain: str
    metrics: List[str]
    dimensions: List[str]
    filters: List[str]


@dataclass
class ContextPackage:
    domain: str
    tables: List[Dict[str, Any]]
    joins: List[Dict[str, Any]]
    rules: List[str]
    examples: List[Dict[str, Any]]
    context_text: str


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9_]+", text.lower())


def normalize_intent(question: str) -> NormalizedIntent:
    q = question.strip()
    q_low = q.lower()
    domain = "ventas"

    for candidate, words in BUSINESS_GLOSSARY["domains"].items():
        if any(word in q_low for word in words):
            domain = candidate
            break

    metrics: List[str] = []
    dimensions: List[str] = []
    for word, cols in BUSINESS_GLOSSARY["metrics"].items():
        if word in q_low:
            metrics.extend(cols)
    for word, cols in BUSINESS_GLOSSARY["dimensions"].items():
        if word in q_low:
            dimensions.extend(cols)

    filters: List[str] = []
    years = re.findall(r"\b20\d{2}\b", q_low)
    for year in years:
        if domain == "finanzas":
            filters.append(f"ANIO_FISCAL={year[2:]}")
        else:
            filters.append(f"YEAR(FECHA)={year}")

    return NormalizedIntent(
        question=q,
        domain=domain,
        metrics=sorted(set(metrics)),
        dimensions=sorted(set(dimensions)),
        filters=filters,
    )


def _score_table(intent: NormalizedIntent, table: Dict[str, Any]) -> int:
    score = 0
    q_tokens = set(_tokenize(intent.question))
    tname = table["table_name"].lower()
    tdesc = table.get("description", "").lower()

    if intent.domain in str(table.get("domain", "")):
        score += 10

    target = {c.lower() for c in intent.metrics + intent.dimensions}
    for col in table["columns"]:
        cname = col["name"].lower()
        if cname in target:
            score += 8
        for token in q_tokens:
            if token in cname:
                score += 2

    for token in q_tokens:
        if token in tname:
            score += 3
        if token in tdesc:
            score += 1

    return score


def _compress_columns(intent: NormalizedIntent, table: Dict[str, Any], max_cols: int = 8) -> List[str]:
    q_tokens = set(_tokenize(intent.question))
    scored: List[Tuple[int, str]] = []
    target = {c.lower() for c in intent.metrics + intent.dimensions}

    for col in table["columns"]:
        cname = col["name"]
        lcname = cname.lower()
        score = 0
        if lcname in target:
            score += 10
        for token in q_tokens:
            if token in lcname:
                score += 3
        if lcname in {"date_conta","fecha","mes","id_cliente","id_producto","id_compania","ccia","cia"}:
            score += 2
        scored.append((score, cname))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [name for _, name in scored[:max_cols]]


def select_relevant_context(intent: NormalizedIntent, max_tables: int = 4) -> ContextPackage:
    ranked = sorted(SCHEMA_CATALOG["tables"], key=lambda t: _score_table(intent, t), reverse=True)
    tables = [t for t in ranked if _score_table(intent, t) > 0][:max_tables]
    if not tables:
        tables = ranked[:2]

    table_names = {t["table_name"] for t in tables}
    joins = [j for j in JOIN_WHITELIST["joins"] if j["left_table"] in table_names or j["right_table"] in table_names][:4]
    rules = [r["rule"] for r in BUSINESS_RULES["rules"] if r["domain"] in {"global", intent.domain}][:5]
    examples = [e for e in SQL_EXAMPLES if e["domain"] == intent.domain][:1]

    table_blocks = []
    for t in tables:
        short = t["table_name"].replace("DDM_ERP.", "")
        cols = ",".join(_compress_columns(intent, t))
        table_blocks.append(f"{short}[{cols}]")

    join_blocks = []
    for j in joins:
        lt = j["left_table"].replace("DDM_ERP.", "")
        rt = j["right_table"].replace("DDM_ERP.", "")
        cond = j["condition"].replace("DDM_ERP.", "")
        join_blocks.append(f"{lt}->{rt}:{cond}")

    example_block = f"Ej:{examples[0]['sql'][:180]}" if examples else ""
    context_text = f"dom={intent.domain}; tbl={' | '.join(table_blocks)}; join={' | '.join(join_blocks)}; rules={' | '.join(rules)}; {example_block}"
    return ContextPackage(intent.domain, tables, joins, rules, examples, context_text)


def build_prompt(question: str, context: ContextPackage, max_chars: int = 1500) -> str:
    base = "Genera solo SQL MySQL/SingleStore. Solo SELECT. No inventes tablas, columnas ni joins. Usa solo el contexto. Si no alcanza, responde NO_SQL. Agrega LIMIT 1000 si falta."
    prompt = f"{base}\nCTX:{context.context_text}\nQ:{question}\nSQL:"
    if len(prompt) <= max_chars:
        return prompt

    compact = re.sub(r"Ej:.*$", "", context.context_text).strip()
    prompt = f"{base}\nCTX:{compact}\nQ:{question}\nSQL:"
    if len(prompt) <= max_chars:
        return prompt

    table_names = ",".join(t["table_name"].replace("DDM_ERP.", "") for t in context.tables)
    join_names = " | ".join(f"{j['left_table'].replace('DDM_ERP.','')}={j['right_table'].replace('DDM_ERP.','')}" for j in context.joins[:3])
    ultra = f"dom={context.domain}; tbl={table_names}; join={join_names}; rules=solo SELECT,no inventar,si no alcanza NO_SQL"
    return f"{base}\nCTX:{ultra}\nQ:{question}\nSQL:"[:max_chars]


def build_client(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def generate_sql(question: str, model: str = "qwen2.5-coder:3b", client: Optional[object] = None, temperature: float = 0.0) -> Dict[str, Any]:
    client = client or build_client()
    intent = normalize_intent(question)
    context = select_relevant_context(intent)
    prompt = build_prompt(question, context, max_chars=1500)

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "Devuelve únicamente SQL o NO_SQL."},
            {"role": "user", "content": prompt}
        ]
    )
    sql = response.choices[0].message.content.strip()
    return {
        "question": question,
        "intent": intent.__dict__,
        "context": context.context_text,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "sql": sql
    }


def _extract_tables(sql: str) -> List[str]:
    tables = re.findall(r"(?:FROM|JOIN)\s+([A-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
    output = []
    for t in tables:
        t = t.upper()
        if "." not in t:
            t = f"DDM_ERP.{t}"
        if t not in output:
            output.append(t)
    return output


def _extract_columns(sql: str) -> List[str]:
    refs = re.findall(r"\b([A-Z][A-Z0-9_]*)\.([A-Z][A-Z0-9_]*)\b", sql, flags=re.IGNORECASE)
    output: List[str] = []
    for prefix, col in refs:
        if prefix.upper() == "DDM_ERP":
            continue
        output.append(col.upper())
    return output


def validate_sql(sql: str) -> Dict[str, Any]:
    errors: List[str] = []
    raw = sql.strip()

    if raw == "NO_SQL":
        return {"valid": False, "errors": ["LLM devolvió NO_SQL"], "sql": raw}

    if not re.match(r"^\s*SELECT\b", raw, flags=re.IGNORECASE):
        errors.append("La sentencia no inicia con SELECT.")

    blocked = re.findall(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b", raw, flags=re.IGNORECASE)
    if blocked:
        errors.append(f"Palabras bloqueadas detectadas: {sorted(set(x.upper() for x in blocked))}")

    if raw.count(";") > 1:
        errors.append("Múltiples sentencias detectadas.")
    if "--" in raw or "/*" in raw:
        errors.append("No se permiten comentarios SQL.")

    allowed_tables = {t["table_name"].upper() for t in SCHEMA_CATALOG["tables"]}
    tables = _extract_tables(raw)
    bad_tables = [t for t in tables if t not in allowed_tables]
    if bad_tables:
        errors.append(f"Tablas no permitidas: {bad_tables}")

    allowed_columns = {str(c["name"]).upper() for t in SCHEMA_CATALOG["tables"] for c in t["columns"]}
    columns = _extract_columns(raw)
    bad_columns = [c for c in columns if c not in allowed_columns]
    if bad_columns:
        errors.append(f"Columnas no permitidas: {sorted(set(bad_columns))[:10]}")

    if " LIMIT " not in f" {raw.upper()} ":
        errors.append("Falta LIMIT 1000 o un límite explícito.")

    return {"valid": len(errors) == 0, "errors": errors, "tables": tables, "columns": columns[:20], "sql": raw}


def repair_sql(question: str, bad_sql: str, validation_errors: List[str], model: str = "qwen2.5-coder:3b", client: Optional[object] = None) -> str:
    client = client or build_client()
    intent = normalize_intent(question)
    context = select_relevant_context(intent)
    prompt = (
        "Corrige el SQL usando solo el contexto. Solo SELECT. Si no puedes corregir, devuelve NO_SQL. "
        f"Errores:{' | '.join(validation_errors)} "
        f"CTX:{context.context_text} "
        f"SQL_Original:{bad_sql} "
        f"Q:{question} SQL:"
    )[:1500]

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Devuelve únicamente SQL corregido o NO_SQL."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    question = "ventas por mes y categoria contable en 2025"
    result = generate_sql(question)
    print(result["prompt_chars"])
    print(result["prompt"])
    print(result["sql"])
    print(validate_sql(result["sql"]))
