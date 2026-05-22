"""
CoopTech Backend — Agente 7: Validación de Edad.

Bloqueo estricto a menores de 18 años.
Validación cruzada entre campo 'edad' y 'v_fecha_nac'.
Detección de inconsistencias.
"""

import asyncio
import logging
from datetime import datetime

import pandas as pd

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AgeValidationAgent(BaseAgent):
    """
    Agente de validación de edad con bloqueo estricto.
    Verifica consistencia entre campos de edad y fecha de nacimiento.
    """

    def __init__(self):
        super().__init__(
            agent_id="age_validation",
            agent_name="Validación de Edad",
        )
        self.validation_results: pd.DataFrame = pd.DataFrame()
        self.blocked_minors: list[dict] = []
        self.inconsistencies: list[dict] = []

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Ejecuta validación de edad para todos los socios."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._validate_ages, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _validate_ages(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Valida edad, detecta menores e inconsistencias."""
        # Tomar último período para evitar duplicados
        if "periodo" in df.columns:
            last_periodo = df["periodo"].max()
            df = df[df["periodo"] == last_periodo].copy()

        results_data = []

        for _, row in df.iterrows():
            socio_id = row.get("v_ah_cliente", None)
            edad_campo = row.get("edad", None)
            fecha_nac = row.get("v_fecha_nac", None)
            fecha_proc = row.get("fecha_proceso", None)
            menor_campo = row.get("menor_edad", None)

            # Recalcular edad desde fecha de nacimiento
            edad_calculada = None
            if pd.notna(fecha_nac) and pd.notna(fecha_proc):
                try:
                    delta = fecha_proc - fecha_nac
                    edad_calculada = round(delta.days / 365.25, 1)
                except Exception:
                    pass

            # Determinar edad real
            edad_real = edad_calculada if edad_calculada is not None else edad_campo

            # Flags
            is_minor = bool(edad_real is not None and edad_real < 18)
            is_inconsistent = False
            inconsistency_type = None

            if edad_campo is not None and edad_calculada is not None:
                diff = abs(edad_campo - edad_calculada)
                if diff > 2:
                    is_inconsistent = True
                    inconsistency_type = f"Campo dice {edad_campo}, calculada {edad_calculada:.1f} (diff={diff:.1f})"

            # Inconsistencia campo menor_edad
            if menor_campo is not None and edad_real is not None:
                expected_menor = 1 if edad_real < 18 else 0
                if int(menor_campo) != expected_menor:
                    is_inconsistent = True
                    inconsistency_type = (inconsistency_type or "") + f" | menor_edad={menor_campo} pero edad={edad_real}"

            results_data.append({
                "v_ah_cliente": socio_id,
                "edad_campo": edad_campo,
                "edad_calculada": edad_calculada,
                "edad_real": edad_real,
                "is_minor": is_minor,
                "is_blocked": is_minor,  # Bloqueo estricto = todos los menores
                "is_inconsistent": is_inconsistent,
                "inconsistency_detail": inconsistency_type,
            })

        self.validation_results = pd.DataFrame(results_data)

        # Listas de menores bloqueados e inconsistencias
        self.blocked_minors = (
            self.validation_results[self.validation_results["is_blocked"]]
            [["v_ah_cliente", "edad_real", "edad_campo", "edad_calculada"]]
            .to_dict("records")
        )

        self.inconsistencies = (
            self.validation_results[self.validation_results["is_inconsistent"]]
            [["v_ah_cliente", "edad_campo", "edad_calculada", "inconsistency_detail"]]
            .to_dict("records")
        )

        # Distribución etaria
        edad_series = self.validation_results["edad_real"].dropna()
        age_distribution = {}
        if len(edad_series) > 0:
            bins = [0, 18, 25, 35, 45, 55, 65, 75, 150]
            labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
            age_groups = pd.cut(edad_series, bins=bins, labels=labels, right=False)
            age_distribution = age_groups.value_counts().to_dict()
            age_distribution = {str(k): int(v) for k, v in age_distribution.items()}

        metrics = {
            "total_socios": len(df),
            "menores_bloqueados": len(self.blocked_minors),
            "pct_menores": round(len(self.blocked_minors) / len(df) * 100, 2) if len(df) > 0 else 0,
            "inconsistencias_detectadas": len(self.inconsistencies),
            "pct_inconsistencias": round(len(self.inconsistencies) / len(df) * 100, 2) if len(df) > 0 else 0,
            "edad_promedio": round(float(edad_series.mean()), 1) if len(edad_series) > 0 else 0,
            "edad_mediana": round(float(edad_series.median()), 1) if len(edad_series) > 0 else 0,
        }

        results = {
            "blocked_minors": self.blocked_minors[:50],
            "inconsistencies": self.inconsistencies[:50],
            "age_distribution": age_distribution,
            "minor_client_ids": [m["v_ah_cliente"] for m in self.blocked_minors],
        }

        return metrics, results

    async def predict(self, input_data: dict) -> dict:
        """Valida la edad de un socio individual."""
        edad = input_data.get("edad")
        fecha_nac = input_data.get("v_fecha_nac")
        fecha_proc = input_data.get("fecha_proceso", datetime.now())

        edad_calculada = None
        if fecha_nac:
            if isinstance(fecha_nac, str):
                fecha_nac = pd.to_datetime(fecha_nac, errors="coerce")
            if isinstance(fecha_proc, str):
                fecha_proc = pd.to_datetime(fecha_proc, errors="coerce")
            if pd.notna(fecha_nac) and pd.notna(fecha_proc):
                edad_calculada = round((fecha_proc - fecha_nac).days / 365.25, 1)

        edad_real = edad_calculada if edad_calculada else edad
        is_minor = edad_real is not None and edad_real < 18

        return {
            "agent_id": self.agent_id,
            "edad_campo": edad,
            "edad_calculada": edad_calculada,
            "edad_real": edad_real,
            "is_minor": is_minor,
            "is_blocked": is_minor,
            "validation_status": "BLOQUEADO - Menor de edad" if is_minor else "APROBADO",
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "age_distribution": self.results.get("age_distribution", {}),
            "n_blocked": len(self.blocked_minors),
            "n_inconsistencies": len(self.inconsistencies),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
