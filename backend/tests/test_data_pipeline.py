"""
CoopTech Backend — Tests del Data Loader y Cleaner.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import _parse_dias_por_mes, _parse_dates_flexible
from data.cleaner import (
    clean_column_types,
    impute_missing_values,
    validate_age_consistency,
    handle_outliers,
    encode_categoricals,
    deduplicate,
    run_full_cleaning,
)


# ─── Tests del Loader ────────────────────────────────────────────────────────

class TestLoader:
    def test_parse_dias_por_mes_valid(self):
        result = _parse_dias_por_mes("{3: 31.0, 4: 30.0}")
        assert result == {3: 31.0, 4: 30.0}

    def test_parse_dias_por_mes_empty(self):
        assert _parse_dias_por_mes("") == {}
        assert _parse_dias_por_mes(None) == {}

    def test_parse_dias_por_mes_invalid(self):
        assert _parse_dias_por_mes("not a dict") == {}

    def test_parse_dates_flexible(self):
        df = pd.DataFrame({
            "fecha_proceso": ["May 1 2026 12:00AM", "Apr 1 2026 12:00AM"],
            "other_col": [1, 2],
        })
        result = _parse_dates_flexible(df, ["fecha_proceso"])
        assert pd.api.types.is_datetime64_any_dtype(result["fecha_proceso"])
        assert result["fecha_proceso"].iloc[0].month == 5

    def test_parse_dates_missing_col(self):
        df = pd.DataFrame({"col_a": [1, 2]})
        result = _parse_dates_flexible(df, ["fecha_inexistente"])
        assert "fecha_inexistente" not in result.columns


# ─── Tests del Cleaner ───────────────────────────────────────────────────────

class TestCleaner:
    @pytest.fixture
    def sample_df(self):
        """DataFrame de muestra para tests."""
        return pd.DataFrame({
            "v_ah_cliente": [100, 200, 300, 400],
            "v_ah_cuenta": [1, 2, 3, 4],
            "saldo_disponible": [1000, "500", 200, 50000],
            "monto_bloq": [100, 0, "50", 0],
            "ingresos": [500, np.nan, 300, 200],
            "egresos": [200, np.nan, 150, 100],
            "edad": [30, 45, "17", 70],
            "sexo": ["M", "F", "m", "F"],
            "estado_cta": ["A", "I", "a", "C"],
            "estado_civil": ["S", "C", "s", "V"],
            "menor_edad": [0, 0, 0, 0],  # edad 17 debería ser 1
            "oficina_cta": [20, 20, 30, 20],
            "prod_bancario": [1, 1, 2, 1],
            "periodo": ["2026-05", "2026-05", "2026-05", "2026-05"],
            "fecha_proceso": pd.to_datetime(["2026-05-01"] * 4),
            "v_fecha_nac": pd.to_datetime(["1996-01-01", "1981-01-01", "2009-01-01", "1956-01-01"]),
        })

    def test_clean_column_types(self, sample_df):
        result = clean_column_types(sample_df)
        assert result["saldo_disponible"].dtype in [np.float64, np.int64, float, int]
        assert result["sexo"].iloc[0] == "M"
        assert result["estado_cta"].iloc[2] == "A"

    def test_impute_missing_ingresos(self, sample_df):
        sample_df = clean_column_types(sample_df)
        result = impute_missing_values(sample_df)
        assert result["ingresos"].isna().sum() == 0

    def test_impute_missing_egresos(self, sample_df):
        sample_df = clean_column_types(sample_df)
        result = impute_missing_values(sample_df)
        assert result["egresos"].isna().sum() == 0

    def test_validate_age_consistency(self, sample_df):
        result = validate_age_consistency(sample_df)
        # Socio con edad 17 debería tener menor_edad = 1
        assert result.loc[result["v_ah_cliente"] == 300, "menor_edad"].iloc[0] == 1

    def test_handle_outliers(self, sample_df):
        sample_df = clean_column_types(sample_df)
        result = handle_outliers(sample_df, percentile=99)
        # No debería crear NaN
        assert result["saldo_disponible"].isna().sum() == 0

    def test_encode_categoricals(self, sample_df):
        sample_df = clean_column_types(sample_df)
        result = encode_categoricals(sample_df)
        assert "sexo_M" in result.columns
        assert "sexo_F" in result.columns
        assert "estado_cta_label" in result.columns
        assert "ecivil_S" in result.columns

    def test_deduplicate(self, sample_df):
        # Agregar duplicado
        duplicated = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        result = deduplicate(duplicated)
        assert len(result) == len(sample_df)

    def test_run_full_cleaning(self, sample_df):
        result = run_full_cleaning(sample_df)
        assert result["ingresos"].isna().sum() == 0
        assert result["egresos"].isna().sum() == 0
        assert "sexo_M" in result.columns
        assert result.shape[0] <= sample_df.shape[0]


# ─── Tests de Feature Engineering ────────────────────────────────────────────

class TestFeatureEngineering:
    @pytest.fixture
    def clean_df(self):
        """DataFrame limpio para tests de features."""
        return pd.DataFrame({
            "v_ah_cliente": [100, 100, 200, 200],
            "saldo_disponible": [1000, 1200, 500, 400],
            "monto_bloq": [100, 50, 200, 200],
            "ingresos": [500, 600, 300, 250],
            "egresos": [200, 150, 300, 400],
            "edad": [30, 30, 17, 17],
            "credito": [1, 1, 0, 0],
            "cooplinea": [1, 1, 0, 0],
            "tarjetas": [1, 1, 0, 0],
            "v24h": [1, 0, 0, 0],
            "v12h": [0, 0, 0, 0],
            "val_de_creditos": [500, 300, 0, 0],
            "val_de_debitos": [200, 100, 0, 0],
            "certificadosvalor": [100, 150, 0, 0],
            "periodo": ["2026-03", "2026-04", "2026-03", "2026-04"],
            "fecha_proceso": pd.to_datetime(["2026-03-01", "2026-04-01", "2026-03-01", "2026-04-01"]),
            "fecha_aper": pd.to_datetime(["2020-01-01", "2020-01-01", "2024-01-01", "2024-01-01"]),
            "fecha_ultmov": pd.to_datetime(["2026-02-28", "2026-03-31", "2025-12-01", "2025-12-01"]),
            "fecha_actualizacion": pd.to_datetime(["2026-01-01"] * 4),
        })

    def test_financial_features(self, clean_df):
        from data.feature_engineering import create_financial_features
        result = create_financial_features(clean_df)
        assert "ratio_ingreso_egreso" in result.columns
        assert "flujo_neto" in result.columns
        assert "saldo_efectivo" in result.columns
        assert result["flujo_neto"].iloc[0] == 300  # 500 - 200

    def test_temporal_features(self, clean_df):
        from data.feature_engineering import create_temporal_features
        result = create_temporal_features(clean_df)
        assert "antiguedad_cuenta_dias" in result.columns
        assert "dias_sin_movimiento" in result.columns
        assert result["antiguedad_cuenta_dias"].iloc[0] > 0

    def test_demographic_features(self, clean_df):
        from data.feature_engineering import create_demographic_features
        result = create_demographic_features(clean_df)
        assert "es_menor" in result.columns
        assert "productos_digitales" in result.columns
        # Socio con edad 17 → es_menor = 1
        assert result.loc[result["edad"] == 17, "es_menor"].iloc[0] == 1

    def test_trend_features(self, clean_df):
        from data.feature_engineering import create_trend_features
        result = create_trend_features(clean_df)
        assert "delta_saldo_mensual" in result.columns
        assert "tendencia_saldo" in result.columns
