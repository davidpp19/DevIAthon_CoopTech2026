"""
CoopTech Backend — Agente 9: Impacto Familiar.

Ajusta la capacidad de pago estimada basándose en proxies de cargas
familiares derivadas de datos sociodemográficos (estado civil, edad, sexo).
"""

import asyncio
import logging

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class FamilyImpactAgent(BaseAgent):
    """
    Estima el impacto de cargas familiares en la capacidad de pago.
    Opera con proxies sociodemográficos ante la ausencia de datos directos
    de número de hijos o dependientes.
    """

    def __init__(self):
        super().__init__(
            agent_id="family_impact",
            agent_name="Impacto Familiar",
        )
        self.impact_data: pd.DataFrame = pd.DataFrame()

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Calcula impacto familiar para todos los socios."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._compute_family_impact, df, kwargs
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _estimate_dependents(self, row: pd.Series) -> float:
        """
        Estima número probable de dependientes basado en proxies.

        Lógica probabilística:
        - Casado/a → +1 carga (cónyuge)
        - Edad 25-50 → probabilidad de tener hijos (1-3 estimados)
        - Mujer casada 25-45 → mayor probabilidad de hijos
        - Viudo/a > 60 → probablemente sin cargas
        """
        edad = row.get("edad", 0) or 0
        estado_civil = str(row.get("estado_civil", "S")).upper()
        sexo = str(row.get("sexo", "")).upper()

        dependents = 0.0

        # Cónyuge estimado
        if estado_civil in ["C", "U"]:  # Casado o Unión libre
            dependents += 1.0

        # Hijos estimados basados en edad y estado civil
        if estado_civil in ["C", "U", "D", "V"]:
            if 25 <= edad <= 35:
                dependents += 1.5  # Probabilidad media de 1-2 hijos
            elif 35 < edad <= 50:
                dependents += 2.0  # Probabilidad alta de 2-3 hijos
            elif 50 < edad <= 65:
                dependents += 1.0  # Hijos mayores, algunos independientes
            elif edad > 65:
                dependents += 0.3  # Hijos probablemente independientes
        elif estado_civil == "S":
            if 30 <= edad <= 50:
                dependents += 0.5  # Posibilidad baja de dependientes

        return round(dependents, 1)

    def _compute_family_impact(self, df: pd.DataFrame, kwargs: dict) -> tuple[dict, dict]:
        """Calcula factores de ajuste familiar."""
        # Tomar último período
        if "periodo" in df.columns:
            last_periodo = df["periodo"].max()
            df = df[df["periodo"] == last_periodo].copy()

        records = []

        for _, row in df.iterrows():
            socio_id = row.get("v_ah_cliente")
            ingresos = row.get("ingresos", 0) or 0
            egresos = row.get("egresos", 0) or 0

            # Estimar dependientes
            estimated_dependents = self._estimate_dependents(row)

            # Factor de ajuste: cada dependiente reduce capacidad en ~10-15%
            # Factor 1.0 = sin impacto, 0.6 = máximo impacto
            reduction_per_dependent = 0.10
            adjustment_factor = max(0.6, 1.0 - (estimated_dependents * reduction_per_dependent))

            # Ingreso disponible ajustado
            ingreso_ajustado = ingresos * adjustment_factor

            # Score de vulnerabilidad familiar (0 = sin vulnerabilidad, 100 = máxima)
            vulnerability_score = min(100, estimated_dependents * 20 + (
                30 if ingresos < 500 and estimated_dependents > 1 else 0
            ))

            # Capacidad de pago ajustada
            flujo_ajustado = ingreso_ajustado - egresos

            records.append({
                "v_ah_cliente": socio_id,
                "ingresos_original": round(ingresos, 2),
                "estimated_dependents": estimated_dependents,
                "adjustment_factor": round(adjustment_factor, 4),
                "ingreso_ajustado": round(ingreso_ajustado, 2),
                "flujo_ajustado": round(flujo_ajustado, 2),
                "vulnerability_score": round(vulnerability_score, 2),
                "estado_civil": str(row.get("estado_civil", "")),
                "edad": row.get("edad", 0),
                "sexo": str(row.get("sexo", "")),
            })

        self.impact_data = pd.DataFrame(records)

        if self.impact_data.empty:
            return {"total_socios": 0}, {}

        # Estadísticas
        avg_dependents = float(self.impact_data["estimated_dependents"].mean())
        avg_factor = float(self.impact_data["adjustment_factor"].mean())
        avg_vulnerability = float(self.impact_data["vulnerability_score"].mean())
        high_vulnerability = int((self.impact_data["vulnerability_score"] >= 60).sum())

        # Distribución de dependientes estimados
        dep_distribution = {
            "0_deps": int((self.impact_data["estimated_dependents"] == 0).sum()),
            "1_dep": int(((self.impact_data["estimated_dependents"] > 0) &
                          (self.impact_data["estimated_dependents"] <= 1)).sum()),
            "2_deps": int(((self.impact_data["estimated_dependents"] > 1) &
                           (self.impact_data["estimated_dependents"] <= 2)).sum()),
            "3_plus_deps": int((self.impact_data["estimated_dependents"] > 2).sum()),
        }

        # Impacto por estado civil
        impact_by_civil = {}
        for ec in self.impact_data["estado_civil"].unique():
            subset = self.impact_data[self.impact_data["estado_civil"] == ec]
            impact_by_civil[str(ec)] = {
                "n_socios": len(subset),
                "avg_dependents": round(float(subset["estimated_dependents"].mean()), 2),
                "avg_factor": round(float(subset["adjustment_factor"].mean()), 4),
                "avg_vulnerability": round(float(subset["vulnerability_score"].mean()), 2),
            }

        metrics = {
            "total_socios": len(records),
            "avg_estimated_dependents": round(avg_dependents, 2),
            "avg_adjustment_factor": round(avg_factor, 4),
            "avg_vulnerability_score": round(avg_vulnerability, 2),
            "high_vulnerability_count": high_vulnerability,
            "pct_high_vulnerability": round(high_vulnerability / len(records) * 100, 2) if records else 0,
        }

        results = {
            "dependents_distribution": dep_distribution,
            "impact_by_civil_status": impact_by_civil,
            "top_vulnerable": self.impact_data.nlargest(20, "vulnerability_score")[
                ["v_ah_cliente", "vulnerability_score", "estimated_dependents",
                 "adjustment_factor", "ingresos_original", "ingreso_ajustado"]
            ].to_dict("records"),
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Calcula impacto familiar para un socio individual."""
        row = pd.Series(input_data)
        estimated_dependents = self._estimate_dependents(row)
        ingresos = input_data.get("ingresos", 0) or 0

        adjustment_factor = max(0.6, 1.0 - (estimated_dependents * 0.10))
        ingreso_ajustado = ingresos * adjustment_factor
        vulnerability = min(100, estimated_dependents * 20)

        return {
            "agent_id": self.agent_id,
            "estimated_dependents": estimated_dependents,
            "adjustment_factor": round(adjustment_factor, 4),
            "ingreso_original": round(ingresos, 2),
            "ingreso_ajustado": round(ingreso_ajustado, 2),
            "vulnerability_score": round(vulnerability, 2),
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "dependents_distribution": self.results.get("dependents_distribution", {}),
            "impact_by_civil_status": self.results.get("impact_by_civil_status", {}),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
