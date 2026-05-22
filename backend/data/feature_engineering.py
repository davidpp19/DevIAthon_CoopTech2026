"""
CoopTech Backend — Feature Engineering.

Derivación de features calculados a partir de los campos raw para
alimentar los 9 agentes analíticos.
"""

import logging

import numpy as np
import pandas as pd

from config import RISK_PROXY

logger = logging.getLogger(__name__)


def _safe_divide(numerator: pd.Series, denominator: pd.Series, fill: float = 0.0) -> pd.Series:
    """División segura evitando division-by-zero."""
    return numerator / denominator.replace(0, np.nan).fillna(1)


def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features financieros derivados de saldos, ingresos y egresos.
    """
    df = df.copy()

    # Ratio ingreso/egreso — capacidad de pago
    df["ratio_ingreso_egreso"] = _safe_divide(df["ingresos"], df["egresos"] + 1)

    # Flujo neto — liquidez disponible
    df["flujo_neto"] = df["ingresos"] - df["egresos"]

    # Saldo efectivo — fondos realmente utilizables
    df["saldo_efectivo"] = df["saldo_disponible"] - df["monto_bloq"]

    # Ratio de bloqueo — porcentaje de fondos bloqueados
    df["ratio_bloqueo"] = _safe_divide(df["monto_bloq"], df["saldo_disponible"] + 1)

    # Ratio certificados / saldo
    if "certificadosvalor" in df.columns:
        df["ratio_certificados_saldo"] = _safe_divide(
            df["certificadosvalor"], df["saldo_disponible"] + 1
        )
    else:
        df["ratio_certificados_saldo"] = 0.0

    logger.info("Features financieros creados.")
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features temporales: antigüedad, inactividad, frescura de datos.
    """
    df = df.copy()

    # Antigüedad de la cuenta en días
    if "fecha_proceso" in df.columns and "fecha_aper" in df.columns:
        df["antiguedad_cuenta_dias"] = (
            df["fecha_proceso"] - df["fecha_aper"]
        ).dt.days.clip(lower=0).fillna(0)
    else:
        df["antiguedad_cuenta_dias"] = 0

    # Días sin movimiento (inactividad)
    if "fecha_proceso" in df.columns and "fecha_ultmov" in df.columns:
        df["dias_sin_movimiento"] = (
            df["fecha_proceso"] - df["fecha_ultmov"]
        ).dt.days.clip(lower=0).fillna(0)
    else:
        df["dias_sin_movimiento"] = 0

    # Días desde última actualización de datos del socio
    if "fecha_proceso" in df.columns and "fecha_actualizacion" in df.columns:
        df["dias_desde_actualizacion"] = (
            df["fecha_proceso"] - df["fecha_actualizacion"]
        ).dt.days.clip(lower=0).fillna(0)
    else:
        df["dias_desde_actualizacion"] = 0

    logger.info("Features temporales creados.")
    return df


def create_transactional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de actividad transaccional: velocidad, volumen, ratios.
    """
    df = df.copy()

    # Flags de actividad reciente
    df["actividad_24h"] = (df.get("v24h", pd.Series(0, index=df.index)) > 0).astype(int)
    df["actividad_12h"] = (df.get("v12h", pd.Series(0, index=df.index)) > 0).astype(int)

    # Volumen total de movimientos
    val_cred = df.get("val_de_creditos", pd.Series(0, index=df.index))
    val_deb = df.get("val_de_debitos", pd.Series(0, index=df.index))
    df["volumen_total"] = val_cred + val_deb

    # Ratio créditos vs débitos
    df["ratio_creditos_debitos"] = _safe_divide(val_cred, val_deb + 1)

    logger.info("Features transaccionales creados.")
    return df


def create_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features de tendencia cross-período.
    Requiere que el DataFrame tenga columna 'periodo' y datos de múltiples meses.
    """
    df = df.copy()

    if "periodo" not in df.columns or "v_ah_cliente" not in df.columns:
        df["delta_saldo_mensual"] = 0.0
        df["tendencia_saldo"] = 0.0
        logger.info("Features de tendencia: columnas 'periodo' o 'v_ah_cliente' no disponibles.")
        return df

    # Ordenar por cliente y periodo
    df = df.sort_values(["v_ah_cliente", "periodo"])

    # Delta de saldo mensual
    df["delta_saldo_mensual"] = df.groupby("v_ah_cliente")["saldo_disponible"].diff().fillna(0)

    # Tendencia: slope lineal sobre los periodos disponibles
    def _calc_slope(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        x = np.arange(len(group), dtype=float)
        y = group["saldo_disponible"].values.astype(float)
        if np.std(y) == 0:
            return pd.Series(0.0, index=group.index)
        slope = np.polyfit(x, y, 1)[0]
        return pd.Series(slope, index=group.index)

    df["tendencia_saldo"] = df.groupby("v_ah_cliente", group_keys=False).apply(_calc_slope)
    df["tendencia_saldo"] = df["tendencia_saldo"].fillna(0.0)

    logger.info("Features de tendencia creados.")
    return df


def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features demográficos: grupo etario, flags de productos, adopción digital.
    """
    df = df.copy()

    # Grupo etario
    if "edad" in df.columns:
        bins = [0, 18, 30, 45, 60, 75, 150]
        labels = [0, 1, 2, 3, 4, 5]  # 0=menor, 1=joven, 2=adulto, 3=maduro, 4=senior, 5=mayor
        df["grupo_etario"] = pd.cut(df["edad"], bins=bins, labels=labels, right=False)
        df["grupo_etario"] = df["grupo_etario"].astype(float).fillna(0)
        df["es_menor"] = (df["edad"] < 18).astype(int)
    else:
        df["grupo_etario"] = 0
        df["es_menor"] = 0

    # Flags de productos
    df["tiene_credito_activo"] = (df.get("credito", pd.Series(0, index=df.index)) == 1).astype(int)
    df["tiene_tarjeta"] = (df.get("tarjetas", pd.Series(0, index=df.index)) > 0).astype(int)
    df["usa_cooplinea"] = (df.get("cooplinea", pd.Series(0, index=df.index)) > 0).astype(int)

    # Adopción digital
    df["productos_digitales"] = (
        df.get("cooplinea", pd.Series(0, index=df.index)).fillna(0) +
        df.get("tarjetas", pd.Series(0, index=df.index)).fillna(0)
    )

    logger.info("Features demográficos creados.")
    return df


def create_risk_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la variable target proxy para Credit Scoring.

    Un socio se etiqueta como 'riesgo_alto' si cumple:
    - saldo_disponible < percentil 25 del segmento
    - dias_sin_movimiento > 60
    - tiene crédito activo (credito == 1)
    """
    df = df.copy()

    saldo_umbral = df["saldo_disponible"].quantile(
        RISK_PROXY["saldo_umbral_percentil"] / 100
    )

    condicion_saldo = df["saldo_disponible"] < saldo_umbral
    condicion_inactividad = df["dias_sin_movimiento"] > RISK_PROXY["inactividad_dias"]

    if RISK_PROXY["requiere_credito_activo"]:
        condicion_credito = df.get("credito", pd.Series(0, index=df.index)) == 1
    else:
        condicion_credito = pd.Series(True, index=df.index)

    df["riesgo_alto"] = (
        condicion_saldo & condicion_inactividad & condicion_credito
    ).astype(int)

    n_riesgo = df["riesgo_alto"].sum()
    pct_riesgo = n_riesgo / len(df) * 100 if len(df) > 0 else 0
    logger.info(
        f"Target proxy creado: {n_riesgo}/{len(df)} ({pct_riesgo:.1f}%) socios con riesgo alto."
    )

    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de feature engineering.
    Ejecuta todas las derivaciones en orden.

    Args:
        df: DataFrame limpio (post-cleaner).

    Returns:
        DataFrame con todos los features derivados y target proxy.
    """
    logger.info("=" * 60)
    logger.info("INICIO DE FEATURE ENGINEERING")
    logger.info("=" * 60)
    initial_cols = df.shape[1]

    df = create_financial_features(df)
    df = create_temporal_features(df)
    df = create_transactional_features(df)
    df = create_trend_features(df)
    df = create_demographic_features(df)
    df = create_risk_proxy_target(df)

    new_cols = df.shape[1] - initial_cols
    logger.info(f"Feature engineering completo: +{new_cols} features ({initial_cols} → {df.shape[1]} columnas).")
    return df
