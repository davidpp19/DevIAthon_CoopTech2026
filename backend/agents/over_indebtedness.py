"""
CoopTech Backend — Agente 4: Detección de Sobreendeudamiento Oculto.

Identifica socios cuyo flujo neto (ingresos - egresos) es insuficiente
para cubrir obligaciones, evaluando capacidad real de pago.
"""

import asyncio
import logging

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class OverIndebtednessAgent(BaseAgent):
    """
    Detecta sobreendeudamiento oculto comparando flujo neto contra
    indicadores de obligaciones y bloqueos.
    """

    def __init__(self):
        super().__init__(
            agent_id="over_indebtedness",
            agent_name="Detección de Sobreendeudamiento Oculto",
        )
        self.risk_data: pd.DataFrame = pd.DataFrame()

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Analiza sobreendeudamiento para todos los socios."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._compute_overindebtedness, df, kwargs
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _compute_overindebtedness(self, df: pd.DataFrame, kwargs: dict) -> tuple[dict, dict]:
        """Calcula scores de sobreendeudamiento."""
        df = df.copy()

        # Tomar solo el último período para evitar duplicados
        if "periodo" in df.columns:
            last_periodo = df["periodo"].max()
            df = df[df["periodo"] == last_periodo].copy()

        # ── Indicador 1: Ratio ingreso/egreso bajo ──
        ratio_ie = df["ratio_ingreso_egreso"] if "ratio_ingreso_egreso" in df.columns else (
            df["ingresos"] / (df["egresos"] + 1)
        )
        # Normalizar: < 1.0 es crítico, < 1.2 es preocupante
        score_ratio = np.where(
            ratio_ie < 1.0, 1.0,
            np.where(ratio_ie < 1.2, 0.7,
                     np.where(ratio_ie < 1.5, 0.4, 0.1))
        )

        # ── Indicador 2: Ratio de bloqueo alto ──
        ratio_bloq = df["ratio_bloqueo"] if "ratio_bloqueo" in df.columns else (
            df["monto_bloq"] / (df["saldo_disponible"] + 1)
        )
        score_bloqueo = ratio_bloq.clip(upper=1.0)

        # ── Indicador 3: Flujo neto negativo ──
        flujo = df["flujo_neto"] if "flujo_neto" in df.columns else (
            df["ingresos"] - df["egresos"]
        )
        score_flujo = np.where(flujo < 0, 1.0, np.where(flujo < 100, 0.5, 0.1))

        # ── Indicador 4: Tendencia de saldo negativa ──
        if "tendencia_saldo" in df.columns:
            score_tendencia = np.where(
                df["tendencia_saldo"] < -50, 1.0,
                np.where(df["tendencia_saldo"] < 0, 0.5, 0.1)
            )
        else:
            score_tendencia = np.full(len(df), 0.3)

        # ── Indicador 5: Tiene crédito activo (amplifica riesgo) ──
        credit_flag = df.get("credito", pd.Series(0, index=df.index)).fillna(0)
        credit_multiplier = np.where(credit_flag == 1, 1.2, 1.0)

        # ── Score compuesto ──
        composite_score = (
            0.30 * score_ratio +
            0.25 * score_bloqueo +
            0.25 * score_flujo +
            0.20 * score_tendencia
        ) * credit_multiplier

        # Normalizar a 0-100
        composite_score = (composite_score * 100).clip(0, 100)

        # Flag binario
        is_overindebted = (composite_score >= 60).astype(int)

        self.risk_data = pd.DataFrame({
            "v_ah_cliente": df["v_ah_cliente"].values if "v_ah_cliente" in df.columns else df.index,
            "overindebtedness_score": np.round(composite_score, 2),
            "is_overindebted": is_overindebted,
            "flujo_neto": flujo.values,
            "ratio_ingreso_egreso": ratio_ie.values,
            "ratio_bloqueo": ratio_bloq.values,
            "tiene_credito": credit_flag.values,
        })

        # Métricas
        n_over = int(is_overindebted.sum())
        metrics = {
            "total_socios": len(df),
            "socios_sobreendeudados": n_over,
            "pct_sobreendeudados": round(n_over / len(df) * 100, 2) if len(df) > 0 else 0,
            "mean_score": float(np.mean(composite_score)),
            "median_score": float(np.median(composite_score)),
            "socios_flujo_negativo": int((flujo < 0).sum()),
        }

        results = {
            "score_distribution": {
                "bajo_0_30": int((composite_score < 30).sum()),
                "medio_30_60": int(((composite_score >= 30) & (composite_score < 60)).sum()),
                "alto_60_80": int(((composite_score >= 60) & (composite_score < 80)).sum()),
                "critico_80_100": int((composite_score >= 80).sum()),
            },
            "top_risk_socios": self.risk_data.nlargest(20, "overindebtedness_score")[
                ["v_ah_cliente", "overindebtedness_score", "flujo_neto", "ratio_ingreso_egreso"]
            ].to_dict("records"),
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Predicción de sobreendeudamiento para un socio."""
        ingresos = input_data.get("ingresos", 0)
        egresos = input_data.get("egresos", 0)
        monto_bloq = input_data.get("monto_bloq", 0)
        saldo = input_data.get("saldo_disponible", 0)
        credito = input_data.get("credito", 0)

        ratio_ie = ingresos / (egresos + 1)
        flujo = ingresos - egresos
        ratio_bloq = monto_bloq / (saldo + 1)

        score_r = 1.0 if ratio_ie < 1.0 else (0.7 if ratio_ie < 1.2 else 0.1)
        score_b = min(ratio_bloq, 1.0)
        score_f = 1.0 if flujo < 0 else (0.5 if flujo < 100 else 0.1)
        multiplier = 1.2 if credito == 1 else 1.0

        score = (0.30 * score_r + 0.25 * score_b + 0.25 * score_f + 0.20 * 0.3) * multiplier * 100

        return {
            "agent_id": self.agent_id,
            "overindebtedness_score": round(min(score, 100), 2),
            "is_overindebted": score >= 60,
            "flujo_neto": flujo,
            "ratio_ingreso_egreso": round(ratio_ie, 4),
            "risk_level": "crítico" if score >= 80 else "alto" if score >= 60 else "medio" if score >= 30 else "bajo",
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "score_distribution": self.results.get("score_distribution", {}),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
