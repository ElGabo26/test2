from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_DIR = BASE_DIR / "catalog"
EXAMPLES_DIR = BASE_DIR / "examples"
MAX_PROMPT_CHARS = 2500
DEFAULT_DATE_START = "2025-01-01"
BUSINESS_NAME_PREFIXES = ("NOM_", "DESC_")
CURRENT_DATE = date.today().isoformat()
STOPWORDS = {
    "de", "la", "el", "los", "las", "del", "por", "para", "y", "o", "en", "a", "un", "una",
    "con", "sin", "que", "cuanto", "cuántos", "cual", "cuál", "mes", "año", "fecha", "hasta",
    "desde", "hoy", "actual", "totales", "total", "ventas", "venta", "mostrar", "dame", "quiero"
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


SCHEMA_CATALOG = load_json(CATALOG_DIR / "schema_catalog.json")
JOIN_WHITELIST = load_json(CATALOG_DIR / "join_whitelist.json")
BUSINESS_RULES = load_json(CATALOG_DIR / "business_rules.json")
BUSINESS_GLOSSARY = load_json(CATALOG_DIR / "business_glossary.json")
TABLES_BY_NAME = {t["table_name"]: t for t in SCHEMA_CATALOG["tables"]}


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
    date_filter_sql: str
    date_filter_label: str
    requires_dim_fecha: bool = True


@dataclass
class ContextPackage:
    domain: str
    tables: List[Dict[str, Any]]
    joins: List[Dict[str, Any]]
    rules: List[str]
    examples: List[Dict[str, Any]]
    context_text: str
    selected_table_names: List[str]
    preferred_name_columns: Dict[str, List[str]]


@dataclass
class JoinUsage:
    left_table: str
    right_table: str
    join_type: str
    condition: str


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9_]+", text.lower())


def _question_tokens(question: str) -> List[str]:
    tokens = [t for t in _tokenize(question) if len(t) > 2 and t not in STOPWORDS and not re.fullmatch(r"20\d{2}", t)]
    return tokens


def _domain_from_question(q_low: str) -> str:
    domain = "ventas"
    for candidate, words in BUSINESS_GLOSSARY["domains"].items():
        if any(word in q_low for word in words):
            domain = candidate
            break
    return domain


def _normalize_date_filter(q_low: str) -> Tuple[str, str]:
    full_dates = re.findall(r"\b20\d{2}-\d{2}-\d{2}\b", q_low)
    years = re.findall(r"\b20\d{2}\b", q_low)

    if len(full_dates) >= 2:
        start, end = sorted(full_dates[:2])
        return f"DF.FECHA BETWEEN '{start}' AND '{end}'", f"rango explícito {start} a {end}"
    if len(full_dates) == 1:
        one = full_dates[0]
        return f"DF.FECHA = '{one}'", f"fecha explícita {one}"
    if years:
        year = years[0]
        end = CURRENT_DATE if CURRENT_DATE.startswith(year) else f"{year}-12-31"
        return f"DF.FECHA BETWEEN '{year}-01-01' AND '{end}'", f"año explícito {year}"
    return f"DF.FECHA BETWEEN '{DEFAULT_DATE_START}' AND '{CURRENT_DATE}'", f"rango por defecto {DEFAULT_DATE_START} a {CURRENT_DATE}"


def normalize_intent(question: str) -> NormalizedIntent:
    q = question.strip()
    q_low = q.lower()
    domain = _domain_from_question(q_low)

    metrics: List[str] = []
    dimensions: List[str] = []
    for word, cols in BUSINESS_GLOSSARY["metrics"].items():
        if word in q_low:
            metrics.extend(cols)
    for word, cols in BUSINESS_GLOSSARY["dimensions"].items():
        if word in q_low:
            dimensions.extend(cols)

    if not metrics and domain == "ventas":
        metrics.append("VENTA_AUTOCONSUMO")

    date_filter_sql, date_filter_label = _normalize_date_filter(q_low)
    filters = [date_filter_label, "Siempre usar JOIN con DIM_FECHA para filtrar fechas"]

    return NormalizedIntent(
        question=q,
        domain=domain,
        metrics=sorted(set(metrics)),
        dimensions=sorted(set(dimensions)),
        filters=filters,
        date_filter_sql=date_filter_sql,
        date_filter_label=date_filter_label,
    )


def _is_name_column(col_name: str) -> bool:
    return col_name.upper().startswith(BUSINESS_NAME_PREFIXES)


def _name_columns(table: Dict[str, Any], limit: int = 4) -> List[str]:
    cols = [c["name"] for c in table["columns"] if _is_name_column(str(c["name"]))]
    return cols[:limit]


def _score_table(intent: NormalizedIntent, table: Dict[str, Any]) -> int:
    score = 0
    q_tokens = set(_question_tokens(intent.question))
    tname = table["table_name"].lower()
    tdesc = table.get("description", "").lower()

    if intent.requires_dim_fecha and table["table_name"].upper() == "DDM_ERP.DIM_FECHA":
        score += 100
    if intent.domain in str(table.get("domain", "")):
        score += 10
    if any(token in tname for token in q_tokens):
        score += 4
    if any(token in tdesc for token in q_tokens):
        score += 2

    target = {c.lower() for c in intent.metrics + intent.dimensions}
    for col in table["columns"]:
        cname = str(col["name"]).upper()
        lcname = cname.lower()
        if lcname in target:
            score += 8
        for token in q_tokens:
            if token in lcname:
                score += 3
            if token in str(col.get("description", "")).lower():
                score += 1
        if _is_name_column(cname):
            score += 3
            if any(h in lcname for h in ["marca", "canal", "cuenta", "compania", "empresa", "famil", "neg", "ruta", "zona", "agencia", "material", "tipo", "grupo"]):
                score += 2

    return score


def _compress_columns(intent: NormalizedIntent, table: Dict[str, Any], max_cols: int = 10) -> List[str]:
    q_tokens = set(_question_tokens(intent.question))
    target = {c.upper() for c in intent.metrics + intent.dimensions}
    scored: List[Tuple[int, str]] = []

    for col in table["columns"]:
        cname = str(col["name"]).upper()
        lcname = cname.lower()
        score = 0
        if cname in target:
            score += 12
        if table["table_name"].upper() == "DDM_ERP.DIM_FECHA" and cname in {"FECHA", "ANIO", "MES", "MES_NOMBRE", "ANIO_MES", "ID_FECHA"}:
            score += 12
        if _is_name_column(cname):
            score += 6
        for token in q_tokens:
            if token in lcname:
                score += 4
        if cname in {"DATE_CONTA", "ID_CLIENTE", "ID_PRODUCTO", "ID_COMPANIA", "ID_UNID_NEGOCIO", "VENTA_AUTOCONSUMO", "CANT_VENDIDA", "FECHA"}:
            score += 3
        scored.append((score, cname))

    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = [name for _, name in scored[:max_cols]]

    if table["table_name"].upper() != "DDM_ERP.DIM_FECHA":
        for name_col in _name_columns(table, limit=4):
            if name_col not in picked:
                picked.append(name_col)
            if len(picked) >= max_cols:
                break

    return picked[:max_cols]


def _table_by_name(name: str) -> Optional[Dict[str, Any]]:
    return TABLES_BY_NAME.get(name)


def _find_join(left_table: str, right_table: str) -> Optional[Dict[str, Any]]:
    for join in JOIN_WHITELIST["joins"]:
        if join["left_table"] == left_table and join["right_table"] == right_table:
            return join
        if join["left_table"] == right_table and join["right_table"] == left_table:
            return join
    return None


def select_relevant_context(intent: NormalizedIntent, max_tables: int = 5) -> ContextPackage:
    ranked = sorted(SCHEMA_CATALOG["tables"], key=lambda t: _score_table(intent, t), reverse=True)
    selected: List[Dict[str, Any]] = []
    selected_names: List[str] = []

    def add_table(table_name: str) -> None:
        table = _table_by_name(table_name)
        if table and table_name not in selected_names:
            selected.append(table)
            selected_names.append(table_name)

    if intent.domain == "ventas":
        add_table("DDM_ERP.FAC_VENTA_TOTAL")
    if intent.requires_dim_fecha:
        add_table("DDM_ERP.DIM_FECHA")

    for table in ranked:
        if len(selected) >= max_tables:
            break
        if _score_table(intent, table) > 0:
            add_table(table["table_name"])

    if len(selected) < 2:
        for table in ranked[:2]:
            add_table(table["table_name"])

    if intent.requires_dim_fecha and "DDM_ERP.DIM_FECHA" not in selected_names:
        add_table("DDM_ERP.DIM_FECHA")

    joins: List[Dict[str, Any]] = []
    seen_join_keys = set()
    for join in JOIN_WHITELIST["joins"]:
        key = (join["left_table"], join["right_table"], join["condition"])
        if key in seen_join_keys:
            continue
        if join["left_table"] in selected_names and join["right_table"] in selected_names:
            joins.append(join)
            seen_join_keys.add(key)

    if intent.requires_dim_fecha and "DDM_ERP.DIM_FECHA" in selected_names:
        for table_name in list(selected_names):
            if table_name == "DDM_ERP.DIM_FECHA":
                continue
            join = _find_join(table_name, "DDM_ERP.DIM_FECHA")
            if join:
                key = (join["left_table"], join["right_table"], join["condition"])
                if key not in seen_join_keys:
                    joins.insert(0, join)
                    seen_join_keys.add(key)

    rules = [r["rule"] for r in BUSINESS_RULES["rules"] if r["domain"] in {"global", intent.domain}][:6]
    rules.extend([
        "Las columnas que empiezan con NOM_ o DESC_ contienen nombres de negocio. Úsalas para buscar marcas, empresas, unidades, cuentas, canales y también para devolver etiquetas legibles.",
        f"Siempre hacer JOIN con DIM_FECHA y filtrar con {intent.date_filter_sql}.",
    ])
    examples = [e for e in SQL_EXAMPLES if e["domain"] == intent.domain][:2]

    preferred_name_columns = {
        table["table_name"]: _name_columns(table, limit=5)
        for table in selected if _name_columns(table, limit=5)
    }

    table_blocks = []
    for table in selected:
        short_name = table["table_name"].replace("DDM_ERP.", "")
        cols = _compress_columns(intent, table)
        names = [c for c in cols if _is_name_column(c)]
        metric_cols = [c for c in cols if c not in names][:6]
        parts = []
        if metric_cols:
            parts.append("col=" + ",".join(metric_cols[:6]))
        if names:
            parts.append("nom=" + ",".join(names[:4]))
        table_blocks.append(f"{short_name}[{' ; '.join(parts)}]")

    join_blocks = []
    for join in joins[:6]:
        lt = join["left_table"].replace("DDM_ERP.", "")
        rt = join["right_table"].replace("DDM_ERP.", "")
        join_blocks.append(f"{lt}->{rt}:{join['condition']}")

    example_block = ""
    if examples:
        compact_examples = []
        for example in examples:
            compact_examples.append(example["sql"][:220])
        example_block = " ; ".join(compact_examples)

    names_block = " | ".join(
        f"{tbl.replace('DDM_ERP.', '')}:{','.join(cols[:4])}"
        for tbl, cols in preferred_name_columns.items()
    )

    context_text = (
        f"dom={intent.domain}; fechas={intent.date_filter_sql}; "
        f"tablas={' | '.join(table_blocks)}; "
        f"joins={' | '.join(join_blocks)}; "
        f"nom_desc={names_block}; "
        f"reglas={' | '.join(rules[:5])}; "
        f"ej={example_block}"
    )

    return ContextPackage(
        domain=intent.domain,
        tables=selected,
        joins=joins,
        rules=rules,
        examples=examples,
        context_text=context_text,
        selected_table_names=selected_names,
        preferred_name_columns=preferred_name_columns,
    )


def build_prompt(question: str, context: ContextPackage, max_chars: int = MAX_PROMPT_CHARS) -> str:
    base = (
        "Genera solo SQL MySQL/SingleStore. Solo SELECT. Usa únicamente tablas, columnas y joins del CTX. "
        "Siempre usa JOIN con DIM_FECHA para filtrar fechas. Si la pregunta no trae rango, usa DF.FECHA BETWEEN "
        f"'{DEFAULT_DATE_START}' AND '{CURRENT_DATE}'. "
        "Las columnas NOM_* y DESC_* contienen nombres conocidos de negocio; úsalas para buscar términos del usuario y "
        "para devolver nombres legibles en SELECT y GROUP BY. Si no alcanza el contexto, responde NO_SQL. Agrega LIMIT 1000 si falta."
    )
    prompt = f"{base}\nCTX:{context.context_text}\nQ:{question}\nSQL:"
    if len(prompt) <= max_chars:
        return prompt

    compact_context = re.sub(r"; ej=.*$", "", context.context_text).strip()
    prompt = f"{base}\nCTX:{compact_context}\nQ:{question}\nSQL:"
    if len(prompt) <= max_chars:
        return prompt

    reduced_tables = []
    reduced_intent = NormalizedIntent(
        question=question,
        domain=context.domain,
        metrics=[],
        dimensions=[],
        filters=[],
        date_filter_sql=f"DF.FECHA BETWEEN '{DEFAULT_DATE_START}' AND '{CURRENT_DATE}'",
        date_filter_label="reducido",
    )
    for table in context.tables[:4]:
        short_name = table["table_name"].replace("DDM_ERP.", "")
        cols = _compress_columns(reduced_intent, table, max_cols=6)
        reduced_tables.append(f"{short_name}[{','.join(cols[:6])}]")
    reduced_joins = [
        f"{j['left_table'].replace('DDM_ERP.', '')}->{j['right_table'].replace('DDM_ERP.', '')}:{j['condition']}"
        for j in context.joins[:4]
    ]
    ultra = (
        f"dom={context.domain}; fechas=usar DIM_FECHA y DF.FECHA entre '{DEFAULT_DATE_START}' y '{CURRENT_DATE}'; "
        f"tbl={' | '.join(reduced_tables)}; joins={' | '.join(reduced_joins)}; "
        "nom_desc=usar NOM_/DESC_ para buscar y responder; reglas=solo SELECT,no inventar,si no alcanza NO_SQL"
    )
    return f"{base}\nCTX:{ultra}\nQ:{question}\nSQL:"[:max_chars]


def build_client(base_url: str = "http://localhost:11434/v1", api_key: str = "ollama"):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key)


def generate_sql(question: str, model: str = "qwen2.5-coder:3b", client: Optional[object] = None, temperature: float = 0.0) -> Dict[str, Any]:
    client = client or build_client()
    intent = normalize_intent(question)
    context = select_relevant_context(intent)
    prompt = build_prompt(question, context, max_chars=MAX_PROMPT_CHARS)

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
        "selected_tables": context.selected_table_names,
        "preferred_name_columns": context.preferred_name_columns,
        "sql": sql,
    }


def _extract_tables(sql: str) -> List[str]:
    tables = re.findall(r"(?:FROM|JOIN)\s+([A-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
    output = []
    for table in tables:
        value = table.upper()
        if "." not in value:
            value = f"DDM_ERP.{value}"
        if value not in output:
            output.append(value)
    return output


def _extract_columns(sql: str) -> List[str]:
    refs = re.findall(r"\b([A-Z][A-Z0-9_]*)\.([A-Z][A-Z0-9_]*)\b", sql, flags=re.IGNORECASE)
    output: List[str] = []
    for prefix, col in refs:
        if prefix.upper() == "DDM_ERP":
            continue
        output.append(col.upper())
    return output


def _extract_join_usages(sql: str) -> List[JoinUsage]:
    pattern = re.compile(
        r"(LEFT|INNER|RIGHT|FULL)?\s*JOIN\s+([A-Z0-9_\.]+)(?:\s+([A-Z][A-Z0-9_]*))?\s+ON\s+(.+?)(?=\s+(?:LEFT|INNER|RIGHT|FULL)?\s*JOIN\s+|\s+WHERE\s+|\s+GROUP\s+BY\s+|\s+ORDER\s+BY\s+|\s+LIMIT\s+|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    from_match = re.search(r"FROM\s+([A-Z0-9_\.]+)(?:\s+([A-Z][A-Z0-9_]*))?", sql, flags=re.IGNORECASE)
    current_left = from_match.group(1).upper() if from_match else ""
    if current_left and "." not in current_left:
        current_left = f"DDM_ERP.{current_left}"
    usages: List[JoinUsage] = []
    for join_type, right_table, _, condition in pattern.findall(sql):
        rt = right_table.upper()
        if "." not in rt:
            rt = f"DDM_ERP.{rt}"
        usages.append(JoinUsage(
            left_table=current_left,
            right_table=rt,
            join_type=((join_type or "LEFT").upper() + " JOIN"),
            condition=" ".join(condition.split()),
        ))
        current_left = rt
    return usages


def validate_sql(sql: str) -> Dict[str, Any]:
    errors: List[str] = []
    raw = sql.strip()

    if raw == "NO_SQL":
        return {"valid": False, "errors": ["LLM devolvió NO_SQL"], "sql": raw, "tables": [], "columns": []}

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

    if "DDM_ERP.DIM_FECHA" not in tables:
        errors.append("Siempre se debe hacer JOIN con DDM_ERP.DIM_FECHA.")
    if not re.search(r"\b(?:DF|DIM_FECHA)\.FECHA\b\s*(?:=|>=|<=|>|<|BETWEEN)", raw, flags=re.IGNORECASE):
        errors.append("El filtro de fechas debe aplicarse sobre DF.FECHA o DIM_FECHA.FECHA.")

    whitelist_conditions = {
        (j["right_table"].upper(), " ".join(j["condition"].upper().split()))
        for j in JOIN_WHITELIST["joins"]
    }
    for usage in _extract_join_usages(raw):
        normalized = (usage.right_table.upper(), " ".join(usage.condition.upper().split()))
        if normalized not in whitelist_conditions:
            errors.append(f"JOIN no permitido: {usage.right_table} ON {usage.condition}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "tables": tables,
        "columns": columns[:30],
        "sql": raw,
    }


def repair_sql(question: str, bad_sql: str, validation_errors: List[str], model: str = "qwen2.5-coder:3b", client: Optional[object] = None) -> str:
    client = client or build_client()
    intent = normalize_intent(question)
    context = select_relevant_context(intent)
    prompt = (
        "Corrige el SQL usando solo el contexto. Solo SELECT. Siempre usa JOIN con DIM_FECHA y filtra con "
        f"{intent.date_filter_sql}. Usa columnas NOM_/DESC_ para nombres legibles y para buscar términos del usuario. "
        "Si no puedes corregirlo con seguridad, devuelve NO_SQL. "
        f"Errores:{' | '.join(validation_errors)} "
        f"CTX:{context.context_text} "
        f"SQL_Original:{bad_sql} "
        f"Q:{question} SQL:"
    )[:MAX_PROMPT_CHARS]

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
    question = "ventas por mes y canal"
    result = generate_sql(question)
    print(result["prompt_chars"])
    print(result["prompt"])
    print(result["sql"])
    print(validate_sql(result["sql"]))