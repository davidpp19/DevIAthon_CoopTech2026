"""
CoopTech Backend — Cliente Supabase (REST API).

Usa la API REST de Supabase directamente via httpx para
persistir y leer KPIs, segmentos, clusters y datos de socios.
"""

import logging
import json
from typing import Any, Optional
from datetime import datetime

import httpx
import pandas as pd

from supabase_config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TABLE_KPIS,
    TABLE_SEGMENTOS,
    TABLE_CLUSTERS,
    TABLE_SOCIOS,
    TABLE_SCORING,
)

logger = logging.getLogger("supabase-client")

# ─── Headers para la REST API de Supabase ─────────────────────────────────────
_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal",
}

_REST_URL = f"{SUPABASE_URL}/rest/v1"


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _make_serializable(obj: Any) -> Any:
    """Convierte objetos no-serializables a tipos nativos de Python."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, float) and (pd.isna(obj)):
        return None
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return obj


def _clean_dict(d: dict) -> dict:
    """Limpia un dict para serialización JSON."""
    return {k: _make_serializable(v) for k, v in d.items()}


# ─── Upsert genérico ─────────────────────────────────────────────────────────
async def _upsert(table: str, data: list[dict], on_conflict: str = "id") -> bool:
    """Inserta o actualiza registros en una tabla de Supabase."""
    if not data:
        return True

    url = f"{_REST_URL}/{table}"
    headers = {**_HEADERS, "Prefer": "resolution=merge-duplicates,return=minimal"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Enviar en chunks de 500
            chunk_size = 500
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                # Limpiar cada registro
                clean_chunk = [_clean_dict(row) for row in chunk]
                resp = await client.post(url, json=clean_chunk, headers=headers)
                if resp.status_code not in (200, 201, 204):
                    logger.error(
                        "Supabase upsert error [%s]: %s — %s",
                        table,
                        resp.status_code,
                        resp.text[:500],
                    )
                    return False
        logger.info("Supabase: %d registros subidos a '%s'.", len(data), table)
        return True
    except Exception as e:
        logger.error("Supabase upsert exception [%s]: %s", table, e)
        return False


async def _select(table: str, params: Optional[dict] = None) -> list[dict]:
    """Lee registros de una tabla de Supabase."""
    url = f"{_REST_URL}/{table}"
    headers = {**_HEADERS, "Prefer": "return=representation"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers, params=params or {})
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                "Supabase select error [%s]: %s — %s",
                table,
                resp.status_code,
                resp.text[:300],
            )
            return []
    except Exception as e:
        logger.warning("Supabase select exception [%s]: %s", table, e)
        return []


# ─── Funciones públicas ──────────────────────────────────────────────────────

async def upload_kpis(kpis: dict) -> bool:
    """Persiste los KPIs del pipeline en Supabase."""
    # Solo enviar columnas que existen en la tabla
    allowed_keys = {
        "pipeline_status", "total_socios", "tasa_mora_pct", "tasa_riesgo_alto",
        "socios_en_alerta", "socios_sobreendeudados", "menores_bloqueados",
        "menores_detectados", "alertas_desvio", "clusters_count",
        "avg_credit_score", "tasa_imputacion_ingresos_pct",
        "distribucion_riesgo", "distribucion_alerta", "distribucion_canal",
    }
    record = {
        "id": "latest",
        "updated_at": datetime.utcnow().isoformat(),
    }
    for k, v in kpis.items():
        if k in allowed_keys:
            record[k] = _make_serializable(v)
    # Convertir dicts anidados a JSON strings
    for key in ("distribucion_riesgo", "distribucion_alerta", "distribucion_canal"):
        if key in record and isinstance(record[key], dict):
            record[key] = json.dumps(record[key])
    return await _upsert(TABLE_KPIS, [record], on_conflict="id")


async def upload_segmentos(segmentos: list[dict]) -> bool:
    """Persiste los segmentos por producto en Supabase."""
    records = []
    for seg in segmentos:
        records.append({
            "id": f"prod_{seg.get('prod_bancario', 'unknown')}",
            "updated_at": datetime.utcnow().isoformat(),
            **_clean_dict(seg),
        })
    return await _upsert(TABLE_SEGMENTOS, records, on_conflict="id")


async def upload_clusters(clusters: dict) -> bool:
    """Persiste la distribución de clusters en Supabase."""
    records = []
    for name, count in clusters.items():
        records.append({
            "id": name,
            "cluster_name": name,
            "n_socios": int(count),
            "updated_at": datetime.utcnow().isoformat(),
        })
    return await _upsert(TABLE_CLUSTERS, records, on_conflict="id")


async def upload_scoring_result(result: dict) -> bool:
    """Persiste el resultado de un scoring individual en Supabase."""
    cliente_id = result.get("cliente_id")
    if cliente_id is None:
        return False

    record = {
        "id": str(cliente_id),
        "updated_at": datetime.utcnow().isoformat(),
        "cliente_id": cliente_id,
        "riesgo_global": result.get("riesgo_global", 0),
        "elegibilidad_credito": result.get("elegibilidad_credito", False),
        "canal_cobranza": result.get("canal_cobranza"),
        "dia_pago_sugerido": result.get("dia_pago_sugerido"),
        "alertas_activas": json.dumps(result.get("alertas_activas", [])),
        "acciones_priorizadas": json.dumps(result.get("acciones_priorizadas", [])),
        "bloqueos": json.dumps(result.get("bloqueos", [])),
        "agentes": json.dumps(result.get("agentes", {}), default=str),
    }
    return await _upsert(TABLE_SCORING, [record], on_conflict="id")


async def upload_socios_data(df: pd.DataFrame, max_rows: int = 25000) -> bool:
    """Sube datos de socios (último periodo) a Supabase."""
    if df is None or df.empty:
        return False

    # Tomar solo el último periodo
    if "periodo" in df.columns:
        df_last = df[df["periodo"] == df["periodo"].max()].copy()
    else:
        df_last = df.copy()

    # Limitar filas
    if len(df_last) > max_rows:
        df_last = df_last.head(max_rows)

    # Seleccionar columnas clave
    key_cols = [
        "v_ah_cliente", "edad", "sexo", "saldo_disponible", "ingresos",
        "egresos", "credito", "dias_sin_movimiento", "prod_bancario",
        "estado_cta", "oficina_cta", "tipo_cuenta", "desviacion_maxima",
        "fuente_alerta",
    ]
    available_cols = [c for c in key_cols if c in df_last.columns]
    df_sub = df_last[available_cols].copy()

    records = []
    for _, row in df_sub.iterrows():
        d = row.to_dict()
        d["id"] = str(int(d.get("v_ah_cliente", 0)))
        d["updated_at"] = datetime.utcnow().isoformat()
        records.append(_clean_dict(d))

    return await _upsert(TABLE_SOCIOS, records, on_conflict="id")


# ─── Read functions ──────────────────────────────────────────────────────────

async def get_kpis() -> Optional[dict]:
    """Lee los KPIs más recientes de Supabase."""
    rows = await _select(TABLE_KPIS, {"id": "eq.latest"})
    if rows:
        row = rows[0]
        # Parsear JSON strings
        for key in ("distribucion_riesgo", "distribucion_alerta", "distribucion_canal"):
            if key in row and isinstance(row[key], str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
        return row
    return None


async def get_segmentos() -> list[dict]:
    """Lee los segmentos de Supabase."""
    return await _select(TABLE_SEGMENTOS)


async def get_clusters() -> dict:
    """Lee la distribución de clusters de Supabase."""
    rows = await _select(TABLE_CLUSTERS)
    return {r["cluster_name"]: r["n_socios"] for r in rows} if rows else {}


async def get_scoring(cliente_id: str) -> Optional[dict]:
    """Lee el resultado de scoring de un cliente de Supabase."""
    rows = await _select(TABLE_SCORING, {"id": f"eq.{cliente_id}"})
    if rows:
        row = rows[0]
        for key in ("alertas_activas", "acciones_priorizadas", "bloqueos"):
            if key in row and isinstance(row[key], str):
                try:
                    row[key] = json.loads(row[key])
                except json.JSONDecodeError:
                    pass
        if "agentes" in row and isinstance(row["agentes"], str):
            try:
                row["agentes"] = json.loads(row["agentes"])
            except json.JSONDecodeError:
                pass
        return row
    return None


async def test_connection() -> bool:
    """Prueba la conexión a Supabase."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_REST_URL}/",
                headers={**_HEADERS},
            )
            logger.info("Supabase connection test: status=%s", resp.status_code)
            return resp.status_code in (200, 204)
    except Exception as e:
        logger.warning("Supabase connection test failed: %s", e)
        return False
