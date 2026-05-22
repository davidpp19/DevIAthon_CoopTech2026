"""
CoopTech Backend — Agente 2: Sistema de Alerta Temprana.

Clasifica socios en niveles de alerta de morosidad a 30/60/90 días
usando un sistema híbrido (reglas + scoring ponderado).
"""

import asyncio
import logging

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from config import EARLY_WARNING

logger = logging.getLogger(__name__)


class EarlyWarningAgent(BaseAgent):
    """
    Sistema de alerta temprana para morosidad.
    Combina reglas de negocio con scoring ponderado para clasificar
    socios en niveles: Normal, Alerta 30d, Alerta 60d, Alerta 90d.
    """

    def __init__(self):
        super().__init__(
            agent_id="early_warning",
            agent_name="Sistema de Alerta Temprana",
        )
        self.alert_results: pd.DataFrame = pd.DataFrame()

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        Calcula alertas para todos los socios.
        No es ML clásico — es un sistema de reglas ponderado.
        """
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._compute_alerts, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _compute_alerts(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calcula el score de alerta y asigna niveles."""
        df = df.copy()

        # ── Componente 1: Inactividad (peso: 40%) ──
        max_inact = df["dias_sin_movimiento"].quantile(0.99)
        if max_inact == 0:
            max_inact = 1
        inactivity_score = (
            df["dias_sin_movimiento"].clip(upper=max_inact) / max_inact
        )

        # ── Componente 2: Tendencia de saldo (peso: 30%) ──
        # Normalizar tendencia: negativa = peor
        if "tendencia_saldo" in df.columns:
            trend_min = df["tendencia_saldo"].quantile(0.01)
            trend_max = df["tendencia_saldo"].quantile(0.99)
            trend_range = trend_max - trend_min if trend_max != trend_min else 1
            # Invertir: tendencia negativa → score alto
            balance_score = 1 - (
                (df["tendencia_saldo"].clip(trend_min, trend_max) - trend_min) / trend_range
            )
        else:
            balance_score = pd.Series(0.5, index=df.index)

        # ── Componente 3: Desviación (peso: 30%) ──
        if "desviacion_maxima" in df.columns:
            max_desv = df["desviacion_maxima"].quantile(0.99)
            if max_desv == 0:
                max_desv = 1
            deviation_score = (
                df["desviacion_maxima"].clip(upper=max_desv) / max_desv
            )
        else:
            deviation_score = pd.Series(0.0, index=df.index)

        # ── Score compuesto ──
        w = EARLY_WARNING
        alert_score = (
            w["inactivity_weight"] * inactivity_score +
            w["balance_trend_weight"] * balance_score +
            w["deviation_weight"] * deviation_score
        )

        # ── Asignar niveles de alerta ──
        conditions = [
            alert_score >= 0.75,
            alert_score >= 0.50,
            alert_score >= 0.30,
        ]
        choices = ["🔴 Alerta 90d", "🟠 Alerta 60d", "🟡 Alerta 30d"]
        alert_level = np.select(conditions, choices, default="🟢 Normal")

        # ── Estimar días para mora ──
        estimated_days = ((1 - alert_score) * 120).clip(lower=0).round(0).astype(int)

        # Guardar resultados
        self.alert_results = pd.DataFrame({
            "v_ah_cliente": df["v_ah_cliente"] if "v_ah_cliente" in df.columns else df.index,
            "alert_score": alert_score.round(4),
            "alert_level": alert_level,
            "estimated_days_to_default": estimated_days,
            "inactivity_component": inactivity_score.round(4),
            "balance_component": balance_score.round(4),
            "deviation_component": deviation_score.round(4),
        })

        # Métricas
        level_counts = pd.Series(alert_level).value_counts().to_dict()
        metrics = {
            "total_socios": len(df),
            "alert_distribution": level_counts,
            "mean_alert_score": float(alert_score.mean()),
            "median_alert_score": float(alert_score.median()),
            "high_risk_count": int((alert_score >= 0.75).sum()),
            "high_risk_pct": float((alert_score >= 0.75).mean() * 100),
        }

        results = {
            "level_counts": level_counts,
            "top_risk_socios": self.alert_results.nlargest(20, "alert_score")[
                ["v_ah_cliente", "alert_score", "alert_level", "estimated_days_to_default"]
            ].to_dict("records"),
            "score_percentiles": {
                "p25": float(alert_score.quantile(0.25)),
                "p50": float(alert_score.quantile(0.50)),
                "p75": float(alert_score.quantile(0.75)),
                "p90": float(alert_score.quantile(0.90)),
                "p95": float(alert_score.quantile(0.95)),
            },
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Predicción de alerta para un socio individual."""
        if self.alert_results.empty:
            raise RuntimeError("Agente no entrenado. Ejecute train() primero.")

        socio_id = input_data.get("v_ah_cliente")
        if socio_id is not None:
            match = self.alert_results[
                self.alert_results["v_ah_cliente"] == socio_id
            ]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "agent_id": self.agent_id,
                    "v_ah_cliente": socio_id,
                    "alert_score": float(row["alert_score"]),
                    "alert_level": row["alert_level"],
                    "estimated_days_to_default": int(row["estimated_days_to_default"]),
                }

        # Calcular on-the-fly con datos proporcionados
        dias_inact = input_data.get("dias_sin_movimiento", 0)
        tendencia = input_data.get("tendencia_saldo", 0)
        desviacion = input_data.get("desviacion_maxima", 0)

        w = EARLY_WARNING
        score = (
            w["inactivity_weight"] * min(dias_inact / 120, 1) +
            w["balance_trend_weight"] * (0.5 if tendencia >= 0 else 0.8) +
            w["deviation_weight"] * min(desviacion / 30, 1)
        )

        if score >= 0.75:
            level = "🔴 Alerta 90d"
        elif score >= 0.50:
            level = "🟠 Alerta 60d"
        elif score >= 0.30:
            level = "🟡 Alerta 30d"
        else:
            level = "🟢 Normal"

        return {
            "agent_id": self.agent_id,
            "alert_score": round(score, 4),
            "alert_level": level,
            "estimated_days_to_default": int((1 - score) * 120),
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "alert_distribution": self.results.get("level_counts", {}),
            "score_percentiles": self.results.get("score_percentiles", {}),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
