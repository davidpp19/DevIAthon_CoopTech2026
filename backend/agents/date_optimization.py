"""
CoopTech Backend — Agente 8: Optimización de Fechas.

Analiza el desfase entre patrones de ingresos/depósitos y el ciclo
de cuotas para sugerir fechas óptimas de cobro.
"""

import asyncio
import logging

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DateOptimizationAgent(BaseAgent):
    """
    Optimiza fechas de cobro analizando patrones temporales de
    depósitos y movimientos de los socios.
    """

    def __init__(self):
        super().__init__(
            agent_id="date_optimization",
            agent_name="Optimización de Fechas",
        )
        self.optimization_data: pd.DataFrame = pd.DataFrame()

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Analiza patrones de fechas para todos los socios."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._analyze_date_patterns, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _analyze_date_patterns(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Analiza desfases y patrones temporales."""
        df = df.copy()

        records = []

        # Agrupar por socio para análisis cross-período
        if "v_ah_cliente" in df.columns:
            for socio_id, group in df.groupby("v_ah_cliente"):
                group = group.sort_values("fecha_proceso") if "fecha_proceso" in group.columns else group

                # ── Día del mes de último movimiento ──
                if "fecha_ultmov" in group.columns:
                    dias_mov = group["fecha_ultmov"].dropna().dt.day
                    dia_predominante = int(dias_mov.mode().iloc[0]) if len(dias_mov) > 0 and len(dias_mov.mode()) > 0 else 15
                else:
                    dia_predominante = 15

                # ── Análisis de ventanas de actividad (v12h, v24h, v48h) ──
                has_v24h = group.get("v24h", pd.Series(0)).sum() > 0
                has_v12h = group.get("v12h", pd.Series(0)).sum() > 0
                has_v48h = group.get("v48h", pd.Series(0)).sum() > 0

                # Concentración de actividad
                if has_v12h:
                    ventana_preferida = "12h"
                elif has_v24h:
                    ventana_preferida = "24h"
                elif has_v48h:
                    ventana_preferida = "48h"
                else:
                    ventana_preferida = "sin_actividad"

                # ── Desfase estimado ──
                # Si el socio se mueve principalmente a inicio de mes (día 1-10): buen alineamiento
                # Si se mueve a mediados (11-20): desfase medio
                # Si se mueve a final (21-31): posible desfase significativo
                if dia_predominante <= 10:
                    desfase_score = 0.2  # Buen alineamiento
                    fecha_sugerida_cobro = 5
                elif dia_predominante <= 20:
                    desfase_score = 0.5
                    fecha_sugerida_cobro = dia_predominante + 3
                else:
                    desfase_score = 0.8  # Desfase significativo
                    fecha_sugerida_cobro = min(dia_predominante + 5, 28)

                # ── Desviación de patrones (si tiene datos de alerta) ──
                desviacion = float(group["desviacion_maxima"].max()) if "desviacion_maxima" in group.columns else 0

                # Score de alineación (1 = perfectamente alineado, 0 = desalineado)
                alignment_score = round(1.0 - desfase_score * 0.7 - min(desviacion / 30, 1) * 0.3, 4)

                records.append({
                    "v_ah_cliente": socio_id,
                    "dia_movimiento_predominante": dia_predominante,
                    "ventana_actividad": ventana_preferida,
                    "desfase_score": round(desfase_score, 4),
                    "alignment_score": max(0, alignment_score),
                    "fecha_sugerida_cobro": fecha_sugerida_cobro,
                    "desviacion_maxima": desviacion,
                })

        self.optimization_data = pd.DataFrame(records)

        if self.optimization_data.empty:
            metrics = {"total_socios": 0}
            results = {}
            return metrics, results

        # Estadísticas
        avg_alignment = float(self.optimization_data["alignment_score"].mean())
        desalineados = int((self.optimization_data["alignment_score"] < 0.5).sum())

        # Distribución de días sugeridos
        day_distribution = (
            self.optimization_data["fecha_sugerida_cobro"]
            .value_counts()
            .sort_index()
            .to_dict()
        )
        day_distribution = {str(k): int(v) for k, v in day_distribution.items()}

        # Distribución de ventanas de actividad
        window_dist = self.optimization_data["ventana_actividad"].value_counts().to_dict()

        metrics = {
            "total_socios": len(records),
            "avg_alignment_score": round(avg_alignment, 4),
            "socios_desalineados": desalineados,
            "pct_desalineados": round(desalineados / len(records) * 100, 2) if records else 0,
            "dia_cobro_mas_comun": int(self.optimization_data["fecha_sugerida_cobro"].mode().iloc[0]) if len(records) > 0 else 15,
        }

        results = {
            "day_distribution": day_distribution,
            "window_distribution": window_dist,
            "top_desalineados": self.optimization_data.nsmallest(20, "alignment_score")[
                ["v_ah_cliente", "alignment_score", "dia_movimiento_predominante", "fecha_sugerida_cobro"]
            ].to_dict("records"),
            "alignment_percentiles": {
                "p25": float(self.optimization_data["alignment_score"].quantile(0.25)),
                "p50": float(self.optimization_data["alignment_score"].quantile(0.50)),
                "p75": float(self.optimization_data["alignment_score"].quantile(0.75)),
            },
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Sugiere fecha óptima de cobro para un socio."""
        socio_id = input_data.get("v_ah_cliente")

        if not self.optimization_data.empty and socio_id is not None:
            match = self.optimization_data[
                self.optimization_data["v_ah_cliente"] == socio_id
            ]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "agent_id": self.agent_id,
                    "v_ah_cliente": socio_id,
                    "fecha_sugerida_cobro": int(row["fecha_sugerida_cobro"]),
                    "alignment_score": float(row["alignment_score"]),
                    "dia_movimiento_predominante": int(row["dia_movimiento_predominante"]),
                }

        return {
            "agent_id": self.agent_id,
            "fecha_sugerida_cobro": 15,
            "alignment_score": 0.5,
            "message": "Sin datos suficientes. Se sugiere día 15 por defecto.",
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "day_distribution": self.results.get("day_distribution", {}),
            "window_distribution": self.results.get("window_distribution", {}),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
