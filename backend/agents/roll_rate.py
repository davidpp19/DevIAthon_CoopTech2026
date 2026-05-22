"""
CoopTech Backend — Agente 3: Roll-Rate (Predicción de Deterioro de Calificación).

Calcula matrices de transición entre estados a lo largo de períodos
(Marzo → Abril → Mayo) para predecir migraciones futuras.
"""

import asyncio
import logging

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RollRateAgent(BaseAgent):
    """
    Modelo Roll-Rate basado en matrices de transición.
    Analiza cómo los socios migran entre estados (niveles de saldo/actividad)
    entre períodos consecutivos.
    """

    def __init__(self):
        super().__init__(
            agent_id="roll_rate",
            agent_name="Predicción de Deterioro (Roll-Rate)",
        )
        self.transition_matrices: dict = {}
        self.state_distribution: dict = {}

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Calcula matrices de transición cross-período."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._compute_roll_rate, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _assign_state(self, df: pd.DataFrame) -> pd.Series:
        """
        Asigna un estado a cada socio basado en indicadores financieros.
        Estados: 'Excelente', 'Bueno', 'Regular', 'Riesgoso', 'Crítico'
        """
        # Score compuesto basado en saldo efectivo y actividad
        saldo_norm = df["saldo_efectivo"].rank(pct=True) if "saldo_efectivo" in df.columns else 0.5
        if "dias_sin_movimiento" in df.columns:
            # Invertir: menos días inactivo = mejor
            max_dias = df["dias_sin_movimiento"].quantile(0.99)
            if max_dias == 0:
                max_dias = 1
            activity_norm = 1 - (df["dias_sin_movimiento"].clip(upper=max_dias) / max_dias)
        else:
            activity_norm = 0.5

        composite = 0.6 * saldo_norm + 0.4 * activity_norm

        conditions = [
            composite >= 0.8,
            composite >= 0.6,
            composite >= 0.4,
            composite >= 0.2,
        ]
        choices = ["Excelente", "Bueno", "Regular", "Riesgoso"]
        return pd.Series(
            np.select(conditions, choices, default="Crítico"),
            index=df.index,
        )

    def _compute_roll_rate(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Calcula las matrices de transición entre períodos."""
        if "periodo" not in df.columns or "v_ah_cliente" not in df.columns:
            raise ValueError("Se requieren columnas 'periodo' y 'v_ah_cliente'.")

        # Asignar estados
        df = df.copy()
        df["estado_riesgo"] = self._assign_state(df)

        periodos = sorted(df["periodo"].unique())
        states = ["Excelente", "Bueno", "Regular", "Riesgoso", "Crítico"]

        transition_matrices = {}
        socios_at_risk = []

        for i in range(len(periodos) - 1):
            p_from = periodos[i]
            p_to = periodos[i + 1]

            df_from = df[df["periodo"] == p_from][["v_ah_cliente", "estado_riesgo"]].copy()
            df_to = df[df["periodo"] == p_to][["v_ah_cliente", "estado_riesgo"]].copy()

            df_from = df_from.rename(columns={"estado_riesgo": "estado_from"})
            df_to = df_to.rename(columns={"estado_riesgo": "estado_to"})

            merged = df_from.merge(df_to, on="v_ah_cliente", how="inner")

            if merged.empty:
                continue

            # Matriz de transición (porcentajes por fila)
            matrix = pd.crosstab(
                merged["estado_from"],
                merged["estado_to"],
                normalize="index",
            ).reindex(index=states, columns=states, fill_value=0.0)

            transition_key = f"{p_from} → {p_to}"
            transition_matrices[transition_key] = matrix.round(4).to_dict()

            # Socios que deterioraron (migraron a peor estado)
            state_order = {s: i for i, s in enumerate(states)}
            merged["from_idx"] = merged["estado_from"].map(state_order)
            merged["to_idx"] = merged["estado_to"].map(state_order)
            deteriorated = merged[merged["to_idx"] > merged["from_idx"]]

            socios_at_risk.extend(
                deteriorated[["v_ah_cliente", "estado_from", "estado_to"]].head(20).to_dict("records")
            )

        # Distribución actual de estados (último período)
        last_period = periodos[-1]
        df_last = df[df["periodo"] == last_period]
        current_dist = df_last["estado_riesgo"].value_counts().to_dict()

        # Proyección a 1 mes (usar última matriz)
        projected_dist = {}
        if transition_matrices:
            last_matrix_key = list(transition_matrices.keys())[-1]
            last_matrix = pd.DataFrame(transition_matrices[last_matrix_key]).reindex(
                index=states, columns=states, fill_value=0.0
            )
            current_vec = pd.Series(current_dist).reindex(states, fill_value=0)
            current_pct = current_vec / current_vec.sum() if current_vec.sum() > 0 else current_vec
            projected = current_pct.values @ last_matrix.values
            projected_dist = dict(zip(states, np.round(projected * current_vec.sum(), 0).astype(int).tolist()))

        self.transition_matrices = transition_matrices
        self.state_distribution = current_dist

        metrics = {
            "n_periodos": len(periodos),
            "periodos": periodos,
            "n_transiciones": len(transition_matrices),
            "current_distribution": current_dist,
            "projected_distribution_1m": projected_dist,
            "n_socios_deteriorados": len(socios_at_risk),
        }

        results = {
            "transition_matrices": transition_matrices,
            "current_distribution": current_dist,
            "projected_distribution": projected_dist,
            "top_deteriorated_socios": socios_at_risk[:20],
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Predice la probabilidad de migración para un socio."""
        if not self.transition_matrices:
            raise RuntimeError("Agente no entrenado.")

        current_state = input_data.get("estado_riesgo", "Regular")
        states = ["Excelente", "Bueno", "Regular", "Riesgoso", "Crítico"]

        # Usar la última matriz de transición
        last_key = list(self.transition_matrices.keys())[-1]
        matrix = self.transition_matrices[last_key]

        probabilities = matrix.get(current_state, {})

        # Probabilidad de deterioro = P(migrar a estado peor)
        state_idx = states.index(current_state) if current_state in states else 2
        prob_deterioro = sum(
            probabilities.get(s, 0) for s in states[state_idx + 1:]
        )

        return {
            "agent_id": self.agent_id,
            "current_state": current_state,
            "transition_probabilities": probabilities,
            "prob_deterioro": round(prob_deterioro, 4),
            "most_likely_next_state": max(probabilities, key=probabilities.get) if probabilities else current_state,
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "current_distribution": self.state_distribution,
            "n_transition_periods": len(self.transition_matrices),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
