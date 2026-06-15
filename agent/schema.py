"""Schema-rendering helper (provided complete).

Loads the schema directly from sqlite and renders quoted CREATE TABLE
text suitable for prompt context. Identifiers are always double-quoted
so reserved-word table/column names (e.g. `order`) don't break either
the PRAGMA introspection here or the SQL the model emits later.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "bird"


def db_path(db_id: str) -> Path:
    return DB_DIR / f"{db_id}.sqlite"


def _q(ident: str) -> str:
    """Double-quote a SQL identifier, escaping any embedded quotes."""
    return '"' + ident.replace('"', '""') + '"'


# A few distinct example values are shown per text column so the model can tell
# which column holds which kind of value (e.g. that 'Australian Grand Prix'
# lives in races.name, not circuits.name) instead of guessing from the name.
_SAMPLE_LIMIT = 3
_SAMPLE_MAXLEN = 40


def _is_text_column(ctype: str) -> bool:
    # SQLite type affinity: text-ish declared types, plus untyped columns.
    t = ctype.upper()
    return t == "" or any(k in t for k in ("CHAR", "CLOB", "TEXT"))


def _sample_values(
    conn: sqlite3.Connection, table: str, column: str, ctype: str
) -> list[str]:
    """Up to _SAMPLE_LIMIT distinct, non-empty example values for a text column,
    quoted and length-capped. Empty list for non-text columns or on any error."""
    if not _is_text_column(ctype):
        return []
    try:
        rows = conn.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} "
            f"WHERE {_q(column)} IS NOT NULL AND {_q(column)} <> '' "
            f"LIMIT {_SAMPLE_LIMIT}"
        ).fetchall()
    except sqlite3.Error:
        return []
    out: list[str] = []
    for (val,) in rows:
        s = str(val).replace("\n", " ").strip()
        if len(s) > _SAMPLE_MAXLEN:
            s = s[:_SAMPLE_MAXLEN] + "…"
        out.append("'" + s.replace("'", "''") + "'")
    return out


@lru_cache(maxsize=32)
def render_schema(db_id: str) -> str:
    path = db_path(db_id)
    if not path.exists():
        raise FileNotFoundError(
            f"DB {db_id} not found at {path}. Did you run scripts/load_data.py?"
        )

    parts: list[str] = [f"-- Database: {db_id}"]
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        for t in tables:
            parts.append(f"\nCREATE TABLE {_q(t)} (")
            col_lines: list[str] = []
            for _cid, name, ctype, notnull, _dflt, pk in conn.execute(
                f"PRAGMA table_info({_q(t)})"
            ):
                line = f"  {_q(name)} {ctype}"
                if pk:
                    line += " PRIMARY KEY"
                if notnull and not pk:
                    line += " NOT NULL"
                samples = _sample_values(conn, t, name, ctype)
                if samples:
                    line += "  -- e.g. " + ", ".join(samples)
                col_lines.append(line)
            for fk in conn.execute(f"PRAGMA foreign_key_list({_q(t)})"):
                # (id, seq, ref_table, from, to, on_update, on_delete, match)
                col_lines.append(
                    f"  FOREIGN KEY ({_q(fk[3])}) REFERENCES {_q(fk[2])}({_q(fk[4])})"
                )
            parts.append(",\n".join(col_lines))
            parts.append(");")
    return "\n".join(parts)


def available_dbs() -> list[str]:
    if not DB_DIR.exists():
        return []
    return sorted(p.stem for p in DB_DIR.glob("*.sqlite"))
