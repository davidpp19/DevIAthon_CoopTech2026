"""
CoopTech Backend — Agente 6: Análisis de Riesgo por Producto/Destino.

Evaluación estadística de concentración de riesgo agrupada por
producto bancario, tipo de cuenta y oficina.
"""

import asyncio
import logging

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ProductRiskAgent(BaseAgent):
    """
    Analiza el riesgo concentrado por producto bancario, oficina y tipo de cuenta.
    Genera rankings de riesgo y mapas de calor de concentración.
    """

    def __init__(self):
        super().__init__(
            agent_id="product_risk",
            agent_name="Análisis de Riesgo por Producto",
        )
        self.risk_by_product: dict = {}
        self.risk_by_office: dict = {}
        self.risk_by_type: dict = {}

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Calcula métricas de riesgo agrupadas por producto/oficina."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._compute_product_risk, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _compute_group_risk(self, df: pd.DataFrame, group_col: str) -> dict:
        """Calcula métricas de riesgo para un agrupador dado."""
        if group_col not in df.columns:
            return {}

        # Tomar último período
        if "periodo" in df.columns:
            df = df[df["periodo"] == df["periodo"].max()]

        groups = {}
        for name, group in df.groupby(group_col):
            n = len(group)
            if n < 5:
                continue  # Ignorar grupos muy pequeños

            # Tasa de riesgo alto
            tasa_riesgo = float(group["riesgo_alto"].mean()) if "riesgo_alto" in group.columns else 0

            # Saldo promedio
            avg_saldo = float(group["saldo_disponible"].mean()) if "saldo_disponible" in group.columns else 0

            # Inactividad promedio
            avg_inact = float(group["dias_sin_movimiento"].mean()) if "dias_sin_movimiento" in group.columns else 0

            # Concentración (% del total)
            concentracion = round(n / len(df) * 100, 2)

            # Score de riesgo del grupo
            risk_score = (tasa_riesgo * 40 + min(avg_inact / 90, 1) * 30 +
                          (1 - min(avg_saldo / df["saldo_disponible"].quantile(0.75), 1)) * 30
                          if "saldo_disponible" in df.columns else tasa_riesgo * 100)

            groups[str(name)] = {
                "n_socios": n,
                "concentracion_pct": concentracion,
                "tasa_riesgo": round(tasa_riesgo * 100, 2),
                "avg_saldo": round(avg_saldo, 2),
                "avg_dias_inactivo": round(avg_inact, 1),
                "risk_score": round(float(risk_score), 2),
            }

        # Ordenar por risk_score descendente
        groups = dict(sorted(groups.items(), key=lambda x: x[1]["risk_score"], reverse=True))
        return groups

    def _compute_product_risk(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calcula riesgo por las 3 dimensiones."""
        self.risk_by_product = self._compute_group_risk(df, "prod_bancario")
        self.risk_by_office = self._compute_group_risk(df, "oficina_cta")
        self.risk_by_type = self._compute_group_risk(df, "tipo_cuenta")

        # Encontrar los segmentos más riesgosos
        all_segments = []
        for dimension, data in [
            ("producto", self.risk_by_product),
            ("oficina", self.risk_by_office),
            ("tipo_cuenta", self.risk_by_type),
        ]:
            for name, info in data.items():
                all_segments.append({
                    "dimension": dimension,
                    "segment": name,
                    **info,
                })

        all_segments.sort(key=lambda x: x["risk_score"], reverse=True)

        # Índice de concentración HHI (Herfindahl)
        def calc_hhi(risk_dict: dict) -> float:
            if not risk_dict:
                return 0
            total = sum(r["n_socios"] for r in risk_dict.values())
            if total == 0:
                return 0
            shares = [(r["n_socios"] / total) ** 2 for r in risk_dict.values()]
            return round(sum(shares) * 10000, 2)  # HHI en escala 0-10000

        metrics = {
            "n_productos": len(self.risk_by_product),
            "n_oficinas": len(self.risk_by_office),
            "n_tipos_cuenta": len(self.risk_by_type),
            "hhi_producto": calc_hhi(self.risk_by_product),
            "hhi_oficina": calc_hhi(self.risk_by_office),
            "top_risk_segment": all_segments[0] if all_segments else {},
        }

        results = {
            "risk_by_product": self.risk_by_product,
            "risk_by_office": self.risk_by_office,
            "risk_by_type": self.risk_by_type,
            "top_risk_segments": all_segments[:10],
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Retorna el perfil de riesgo del producto/oficina del socio."""
        prod = str(input_data.get("prod_bancario", ""))
        oficina = str(input_data.get("oficina_cta", ""))
        tipo = str(input_data.get("tipo_cuenta", ""))

        return {
            "agent_id": self.agent_id,
            "product_risk": self.risk_by_product.get(prod, {"risk_score": 0, "tasa_riesgo": 0}),
            "office_risk": self.risk_by_office.get(oficina, {"risk_score": 0, "tasa_riesgo": 0}),
            "type_risk": self.risk_by_type.get(tipo, {"risk_score": 0, "tasa_riesgo": 0}),
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "risk_by_product": self.risk_by_product,
            "risk_by_office": self.risk_by_office,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
