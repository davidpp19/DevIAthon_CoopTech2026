"""
CoopTech Backend — Módulo de Limpieza de Datos.

Imputación de nulos, corrección de tipos, encoding de categóricos,
detección de outliers y deduplicación.
"""

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ─── Mapeos de estado ────────────────────────────────────────────────────────
ESTADO_CTA_MAP = {
    "A": "Activa",
    "I": "Inactiva",
    "C": "Cerrada",
}

ESTADO_CIVIL_MAP = {
    "S": "Soltero/a",
    "C": "Casado/a",
    "D": "Divorciado/a",
    "V": "Viudo/a",
    "U": "Unión libre",
}


def clean_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura tipos correctos en columnas numéricas y categóricas.
    """
    df = df.copy()

    # Columnas que deben ser numéricas
    numeric_cols = [
        "saldo_disponible", "monto_bloq", "ult_tasa_int", "tiene_bloqueos",
        "bloqueo_encaje", "v24h", "val_de_creditos", "val_de_debitos",
        "cooplinea", "tarjetas", "credito", "edad", "certificadosvalor",
        "ingresos", "egresos", "menor_edad", "v12h", "v48h", "v72h_difer",
        "int_hoy", "int_acumulado", "saldo_int_decim",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Columnas string limpias
    string_cols = ["estado_cta", "tipo_cuenta", "sexo", "estado_civil", "nacionalidad"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    logger.info("Tipos de columna corregidos.")
    return df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estrategia de imputación:
    - ingresos: mediana por segmento (oficina_cta × prod_bancario)
    - egresos: 0.0 si ingresos > 0, sino mediana segmentada
    - edad: recalcular desde v_fecha_nac si disponible
    - numéricos restantes: 0.0
    """
    df = df.copy()

    # ── Imputar ingresos con mediana por segmento ──
    if "ingresos" in df.columns:
        n_null_ing = df["ingresos"].isna().sum()
        if n_null_ing > 0:
            segment_cols = []
            if "oficina_cta" in df.columns:
                segment_cols.append("oficina_cta")
            if "prod_bancario" in df.columns:
                segment_cols.append("prod_bancario")

            if segment_cols:
                mediana_segmento = df.groupby(segment_cols)["ingresos"].transform("median")
                df["ingresos"] = df["ingresos"].fillna(mediana_segmento)

            # Fallback: mediana global
            df["ingresos"] = df["ingresos"].fillna(df["ingresos"].median())
            # Último fallback
            df["ingresos"] = df["ingresos"].fillna(0.0)
            logger.info(f"Imputados {n_null_ing} valores nulos en 'ingresos'.")

    # ── Imputar egresos ──
    if "egresos" in df.columns:
        n_null_egr = df["egresos"].isna().sum()
        if n_null_egr > 0:
            # Si tiene ingresos > 0 pero egresos nulo → asumir 0 egresos reportados
            mask_ing_positivo = (df["ingresos"] > 0) & df["egresos"].isna()
            df.loc[mask_ing_positivo, "egresos"] = 0.0

            # Resto: mediana segmentada
            if df["egresos"].isna().sum() > 0:
                segment_cols = []
                if "oficina_cta" in df.columns:
                    segment_cols.append("oficina_cta")
                if segment_cols:
                    mediana_egr = df.groupby(segment_cols)["egresos"].transform("median")
                    df["egresos"] = df["egresos"].fillna(mediana_egr)
                df["egresos"] = df["egresos"].fillna(0.0)

            logger.info(f"Imputados {n_null_egr} valores nulos en 'egresos'.")

    # ── Recalcular edad desde fecha de nacimiento ──
    if "v_fecha_nac" in df.columns and "fecha_proceso" in df.columns:
        mask_edad_null = df["edad"].isna()
        if mask_edad_null.any():
            edad_calc = (
                (df["fecha_proceso"] - df["v_fecha_nac"]).dt.days / 365.25
            ).astype(float)
            df.loc[mask_edad_null, "edad"] = edad_calc[mask_edad_null].round(0)
            logger.info(f"Recalculada edad para {mask_edad_null.sum()} registros.")

    # ── Imputar numéricos restantes con 0 ──
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_remaining = df[numeric_cols].isna().sum().sum()
    if n_remaining > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        logger.info(f"Imputados {n_remaining} valores numéricos restantes con 0.0.")

    return df


def validate_age_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valida y corrige inconsistencias entre edad y v_fecha_nac.
    Recalcula menor_edad basado en la edad real.
    """
    df = df.copy()

    # Asegurar que edad sea numérica
    if "edad" in df.columns:
        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")

    if "v_fecha_nac" in df.columns and "fecha_proceso" in df.columns:
        edad_real = ((df["fecha_proceso"] - df["v_fecha_nac"]).dt.days / 365.25).round(0)
        # Detectar inconsistencias (diferencia > 2 años)
        if "edad" in df.columns:
            inconsistentes = (df["edad"].fillna(0) - edad_real).abs() > 2
            n_incons = inconsistentes.sum()
            if n_incons > 0:
                df.loc[inconsistentes, "edad"] = edad_real[inconsistentes]
                logger.warning(f"Corregidas {n_incons} inconsistencias de edad.")

    # Recalcular menor_edad
    if "edad" in df.columns:
        df["menor_edad"] = (df["edad"] < 18).astype(int)

    return df


def handle_outliers(df: pd.DataFrame, percentile: float = 99) -> pd.DataFrame:
    """
    Capping de outliers al percentil especificado para columnas financieras.
    """
    df = df.copy()
    financial_cols = [
        "saldo_disponible", "monto_bloq", "val_de_creditos", "val_de_debitos",
        "ingresos", "egresos", "certificadosvalor",
    ]

    for col in financial_cols:
        if col in df.columns:
            upper = df[col].quantile(percentile / 100)
            n_capped = (df[col] > upper).sum()
            if n_capped > 0:
                df[col] = df[col].clip(upper=upper)
                logger.info(f"Outliers en '{col}': {n_capped} valores capped al p{percentile} ({upper:.2f}).")

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encoding para sexo y estado_civil.
    Label encoding para estado_cta.
    """
    df = df.copy()

    # Label encode estado_cta
    if "estado_cta" in df.columns:
        df["estado_cta_label"] = df["estado_cta"].map({"A": 0, "I": 1, "C": 2}).fillna(-1).astype(int)

    # One-Hot encode sexo
    if "sexo" in df.columns:
        df["sexo_M"] = (df["sexo"] == "M").astype(int)
        df["sexo_F"] = (df["sexo"] == "F").astype(int)

    # One-Hot encode estado_civil
    if "estado_civil" in df.columns:
        for code, label in ESTADO_CIVIL_MAP.items():
            df[f"ecivil_{code}"] = (df["estado_civil"] == code).astype(int)

    logger.info("Variables categóricas codificadas.")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados por (v_ah_cliente, v_ah_cuenta, periodo).
    Mantiene el registro con la fecha de proceso más reciente.
    """
    df = df.copy()
    dedup_keys = ["v_ah_cliente", "v_ah_cuenta", "periodo"]
    existing_keys = [k for k in dedup_keys if k in df.columns]

    if not existing_keys:
        logger.warning("No se encontraron columnas de dedup. Sin deduplicación.")
        return df

    before = df.shape[0]
    if "fecha_proceso" in df.columns:
        df = df.sort_values("fecha_proceso", ascending=False)
    df = df.drop_duplicates(subset=existing_keys, keep="first")
    after = df.shape[0]

    if before > after:
        logger.info(f"Deduplicación: {before} → {after} filas ({before - after} removidas).")
    else:
        logger.info("Sin duplicados encontrados.")

    return df


def run_full_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecuta la pipeline completa de limpieza en orden:
    1. Tipos de columna
    2. Imputación de nulos
    3. Validación de edad
    4. Outliers
    5. Encoding categóricos
    6. Deduplicación

    Args:
        df: DataFrame crudo de la sábana.

    Returns:
        DataFrame limpio y listo para feature engineering.
    """
    logger.info("=" * 60)
    logger.info("INICIO DE LIMPIEZA DE DATOS")
    logger.info("=" * 60)
    initial_shape = df.shape

    df = clean_column_types(df)
    df = impute_missing_values(df)
    df = validate_age_consistency(df)
    df = handle_outliers(df)
    df = encode_categoricals(df)
    df = deduplicate(df)

    logger.info(f"Limpieza completa: {initial_shape} → {df.shape}")
    return df
