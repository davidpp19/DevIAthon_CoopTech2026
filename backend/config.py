"""
CoopTech Backend — Configuración central.
Rutas de datos, hiperparámetros y constantes globales.
"""

from pathlib import Path

# ─── Rutas base ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "DatosAhorrosResumidos"
MODELS_DIR = BASE_DIR / "models"

# Crear directorio de modelos si no existe
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Archivos CSV ────────────────────────────────────────────────────────────
CSV_FILES = {
    "filtrado": {
        "2026-03": DATA_DIR / "Filtrado_Original_DatsSabanaAhorroMarzo1_2026.csv",
        "2026-04": DATA_DIR / "Filtrado_Original_DatsSabanaAhorroAbril1_2026.csv",
        "2026-05": DATA_DIR / "Filtrado_Original_DatsSabanaAhorroMayo1_2026.csv",
    },
    "simplificado": {
        "2026-03": DATA_DIR / "Simplificado_DatsSabanaAhorroMarzo1_2026.csv",
        "2026-04": DATA_DIR / "Simplificado_DatsSabanaAhorroAbril1_2026.csv",
        "2026-05": DATA_DIR / "Simplificado_DatsSabanaAhorroMayo1_2026.csv",
    },
    "alertas": {
        "desviados": DATA_DIR / "alerta_clientes_desviados.csv",
        "desviados_3_mas": DATA_DIR / "alerta_clientes_desviados_3_mas.csv",
    },
}

# ─── Delimitadores por tipo de archivo ────────────────────────────────────────
DELIMITERS = {
    "filtrado": ",",
    "simplificado": ";",
    "alertas": ",",
}

# ─── Columnas de fecha para parsing ──────────────────────────────────────────
DATE_COLUMNS = [
    "fecha_proceso",
    "fecha_aper",
    "fecha_ultmov",
    "fecha_ult_capi",
    "v_fecha_nac",
    "fecha_actualizacion",
]

# ─── Hiperparámetros de modelos ──────────────────────────────────────────────
CREDIT_SCORING = {
    "rf_n_estimators": 200,
    "rf_max_depth": 12,
    "rf_min_samples_split": 10,
    "lr_C": 1.0,
    "lr_max_iter": 1000,
    "test_size": 0.25,
    "random_state": 42,
    "cv_folds": 5,
}

EARLY_WARNING = {
    "threshold_30d": 30,
    "threshold_60d": 60,
    "threshold_90d": 90,
    "inactivity_weight": 0.4,
    "balance_trend_weight": 0.3,
    "deviation_weight": 0.3,
}

CLUSTERING = {
    "k_range": (3, 8),
    "random_state": 42,
    "max_iter": 300,
    "n_init": 10,
}

BALANCER = {
    "smote_k_neighbors": 5,
    "random_state": 42,
}

# ─── Umbrales para variable proxy de riesgo ──────────────────────────────────
RISK_PROXY = {
    "saldo_umbral_percentil": 25,    # Percentil bajo de saldo
    "inactividad_dias": 60,          # Días sin movimiento para flag
    "requiere_credito_activo": True,  # Debe tener crédito = 1
}

# ─── Configuración de la API ─────────────────────────────────────────────────
API_CONFIG = {
    "title": "CoopTech — Predictor de Comportamiento de Pago",
    "description": "Backend analítico con 9 agentes de IA para Coop. Tulcán",
    "version": "1.0.0-mvp",
    "prefix": "/api/v1",
}

# ─── IDs de los 9 agentes ────────────────────────────────────────────────────
AGENT_IDS = {
    "credit_scoring": "Modelo de Credit Scoring",
    "early_warning": "Sistema de Alerta Temprana",
    "roll_rate": "Predicción de Deterioro (Roll-Rate)",
    "over_indebtedness": "Detección de Sobreendeudamiento Oculto",
    "collection_segments": "Segmentación para Cobranza",
    "product_risk": "Análisis de Riesgo por Producto",
    "age_validation": "Validación de Edad",
    "date_optimization": "Optimización de Fechas",
    "family_impact": "Impacto Familiar",
}
