"""
CoopTech Backend — Módulo de Carga de Datos.

Lee los CSV transaccionales y de alertas, unifica periodos en un DataFrame
longitudinal, y detecta delimitadores automáticamente.
"""

import ast
import logging
from pathlib import Path

import pandas as pd

from config import CSV_FILES, DELIMITERS, DATE_COLUMNS

logger = logging.getLogger(__name__)


def _parse_dates_flexible(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """
    Parsea columnas de fecha con formato flexible.
    Maneja el formato 'May 1 2026 12:00AM' y variantes.
    """
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=False, errors="coerce")
    return df


def _parse_dias_por_mes(value: str) -> dict:
    """
    Convierte el campo string '{3: 31.0, 4: 30.0}' a un dict real.
    """
    if pd.isna(value) or not isinstance(value, str):
        return {}
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        logger.warning(f"No se pudo parsear dias_por_mes: {value}")
        return {}


def load_sabana_filtrado() -> pd.DataFrame:
    """
    Carga y concatena los 3 archivos Filtrado_Original (Marzo, Abril, Mayo)
    en un único DataFrame longitudinal con columna 'periodo'.

    Returns:
        DataFrame con ~72K filas (3 meses × ~24K socios) y columna 'periodo'.
    """
    frames = []
    for periodo, filepath in CSV_FILES["filtrado"].items():
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            continue

        logger.info(f"Cargando {filepath.name} (periodo={periodo})...")
        df = pd.read_csv(filepath, delimiter=DELIMITERS["filtrado"], low_memory=False)
        df["periodo"] = periodo
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No se encontraron archivos Filtrado_Original.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _parse_dates_flexible(combined, DATE_COLUMNS)
    logger.info(f"Sábana filtrada cargada: {combined.shape[0]} filas, {combined.shape[1]} columnas.")
    return combined


def load_sabana_simplificado() -> pd.DataFrame:
    """
    Carga y concatena los 3 archivos Simplificado (delimitador ';').
    Socios con menor actividad o subset reducido.

    Returns:
        DataFrame con ~19K filas (3 meses × ~6.3K socios) y columna 'periodo'.
    """
    frames = []
    for periodo, filepath in CSV_FILES["simplificado"].items():
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Archivo no encontrado: {filepath}")
            continue

        logger.info(f"Cargando {filepath.name} (periodo={periodo})...")
        df = pd.read_csv(filepath, delimiter=DELIMITERS["simplificado"], low_memory=False)
        df["periodo"] = periodo
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No se encontraron archivos Simplificado.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _parse_dates_flexible(combined, DATE_COLUMNS)
    logger.info(f"Sábana simplificada cargada: {combined.shape[0]} filas, {combined.shape[1]} columnas.")
    return combined


def load_alertas() -> pd.DataFrame:
    """
    Carga y combina los archivos de alertas de clientes desviados.
    Parsea el campo 'dias_por_mes' de string-dict a features numéricos.

    Returns:
        DataFrame con columnas: cliente, dias_por_mes (dict), desviacion_maxima,
        y features derivados: meses_actividad, max_dias_mes, min_dias_mes, fuente.
    """
    frames = []
    for nombre, filepath in CSV_FILES["alertas"].items():
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Archivo de alertas no encontrado: {filepath}")
            continue

        logger.info(f"Cargando alertas: {filepath.name}...")
        df = pd.read_csv(filepath, delimiter=DELIMITERS["alertas"], low_memory=False)
        df["fuente_alerta"] = nombre  # Diferenciar origen
        frames.append(df)

    if not frames:
        logger.warning("No se encontraron archivos de alertas.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Parsear dias_por_mes y extraer features
    parsed_dicts = combined["dias_por_mes"].apply(_parse_dias_por_mes)
    combined["meses_actividad"] = parsed_dicts.apply(len)
    combined["max_dias_mes"] = parsed_dicts.apply(
        lambda d: max(d.values()) if d else 0
    )
    combined["min_dias_mes"] = parsed_dicts.apply(
        lambda d: min(d.values()) if d else 0
    )

    # Dedup: quedarse con el registro de mayor desviación por cliente
    combined = combined.sort_values("desviacion_maxima", ascending=False)
    combined = combined.drop_duplicates(subset=["cliente"], keep="first")

    logger.info(f"Alertas cargadas: {combined.shape[0]} clientes únicos.")
    return combined


def load_all_data() -> dict[str, pd.DataFrame]:
    """
    Carga todos los datasets disponibles.

    Returns:
        dict con claves 'filtrado', 'simplificado', 'alertas'.
    """
    logger.info("=" * 60)
    logger.info("INICIO DE CARGA DE DATOS")
    logger.info("=" * 60)

    data = {
        "filtrado": load_sabana_filtrado(),
        "simplificado": load_sabana_simplificado(),
        "alertas": load_alertas(),
    }

    total_rows = sum(df.shape[0] for df in data.values())
    logger.info(f"Carga completa. Total: {total_rows} filas en {len(data)} datasets.")
    return data


def merge_with_alerts(
    sabana: pd.DataFrame,
    alertas: pd.DataFrame,
    sabana_key: str = "v_ah_cliente",
    alerta_key: str = "cliente",
) -> pd.DataFrame:
    """
    Enriquece la sábana principal con datos de alertas de desviación.
    Left join para mantener todos los socios (los sin alerta obtienen NaN).

    Args:
        sabana: DataFrame de sábana (filtrado o simplificado).
        alertas: DataFrame de alertas procesadas.
        sabana_key: Columna clave en sábana.
        alerta_key: Columna clave en alertas.

    Returns:
        DataFrame enriquecido con columnas de alerta.
    """
    if alertas.empty:
        logger.warning("DataFrame de alertas vacío. Retornando sábana sin enriquecer.")
        return sabana

    # Seleccionar columnas de alertas relevantes (evitar duplicados)
    alert_cols = [alerta_key, "desviacion_maxima", "meses_actividad",
                  "max_dias_mes", "min_dias_mes", "fuente_alerta"]
    alert_subset = alertas[alert_cols].copy()

    merged = sabana.merge(
        alert_subset,
        left_on=sabana_key,
        right_on=alerta_key,
        how="left",
    )

    # Llenar NaN en columnas de alerta (socios sin alerta = sin desviación)
    merged["desviacion_maxima"] = merged["desviacion_maxima"].fillna(0)
    merged["meses_actividad"] = merged["meses_actividad"].fillna(0)
    merged["max_dias_mes"] = merged["max_dias_mes"].fillna(0)
    merged["min_dias_mes"] = merged["min_dias_mes"].fillna(0)
    merged["fuente_alerta"] = merged["fuente_alerta"].fillna("sin_alerta")

    # Eliminar columna duplicada del join si existe
    if alerta_key in merged.columns and alerta_key != sabana_key:
        merged = merged.drop(columns=[alerta_key])

    n_with_alert = (merged["fuente_alerta"] != "sin_alerta").sum()
    logger.info(f"Merge completado: {n_with_alert}/{merged.shape[0]} filas con alertas.")
    return merged
