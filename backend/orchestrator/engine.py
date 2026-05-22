"""
CoopTech Backend — Motor de Orquestación.

Ejecuta el pipeline completo de datos y los 9 agentes analíticos
en oleadas paralelas con dependencias.
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from data.loader import load_all_data, merge_with_alerts
from data.cleaner import run_full_cleaning
from data.feature_engineering import run_feature_engineering
from data.balancer import prepare_balanced_dataset

from agents.credit_scoring import CreditScoringAgent
from agents.early_warning import EarlyWarningAgent
from agents.roll_rate import RollRateAgent
from agents.over_indebtedness import OverIndebtednessAgent
from agents.collection_segments import CollectionSegmentsAgent
from agents.product_risk import ProductRiskAgent
from agents.age_validation import AgeValidationAgent
from agents.date_optimization import DateOptimizationAgent
from agents.family_impact import FamilyImpactAgent

logger = logging.getLogger(__name__)


class PipelineEngine:
    """
    Orquestador del pipeline completo.
    Ejecuta carga → limpieza → feature engineering → balanceo → 9 agentes.
    """

    def __init__(self):
        # Instanciar los 9 agentes
        self.agents = {
            "credit_scoring": CreditScoringAgent(),
            "early_warning": EarlyWarningAgent(),
            "roll_rate": RollRateAgent(),
            "over_indebtedness": OverIndebtednessAgent(),
            "collection_segments": CollectionSegmentsAgent(),
            "product_risk": ProductRiskAgent(),
            "age_validation": AgeValidationAgent(),
            "date_optimization": DateOptimizationAgent(),
            "family_impact": FamilyImpactAgent(),
        }

        self.status: str = "idle"  # idle, loading, cleaning, engineering, training, ready, error
        self.current_phase: str = ""
        self.df_featured: Optional[pd.DataFrame] = None
        self.balanced_data: Optional[dict] = None
        self.pipeline_start: float = 0
        self.pipeline_duration: float = 0
        self.phase_durations: dict[str, float] = {}

    def get_status(self) -> dict:
        """Retorna el estado actual del pipeline."""
        return {
            "status": self.status,
            "current_phase": self.current_phase,
            "pipeline_duration_seconds": self.pipeline_duration,
            "phase_durations": self.phase_durations,
            "agents": {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            },
        }

    async def run_full_pipeline(self) -> dict:
        """
        Ejecuta el pipeline completo en fases:
        1. Carga de datos
        2. Limpieza e imputación
        3. Feature engineering
        4. Balanceo de clases
        5. Entrenamiento de agentes (3 oleadas paralelas)
        """
        self.pipeline_start = time.time()
        self.status = "running"

        try:
            # ── FASE 1: Carga de datos ──
            self.current_phase = "loading"
            logger.info("=" * 70)
            logger.info("FASE 1: CARGA DE DATOS")
            logger.info("=" * 70)
            t0 = time.time()

            loop = asyncio.get_event_loop()
            raw_data = await loop.run_in_executor(None, load_all_data)
            df_main = raw_data["filtrado"]
            df_alertas = raw_data["alertas"]

            # Merge con alertas
            df_main = await loop.run_in_executor(
                None, merge_with_alerts, df_main, df_alertas
            )

            self.phase_durations["loading"] = round(time.time() - t0, 2)
            logger.info(f"Fase 1 completada en {self.phase_durations['loading']}s — {df_main.shape[0]} registros.")

            # ── FASE 2: Limpieza ──
            self.current_phase = "cleaning"
            logger.info("=" * 70)
            logger.info("FASE 2: LIMPIEZA DE DATOS")
            logger.info("=" * 70)
            t0 = time.time()

            df_clean = await loop.run_in_executor(None, run_full_cleaning, df_main)

            self.phase_durations["cleaning"] = round(time.time() - t0, 2)
            logger.info(f"Fase 2 completada en {self.phase_durations['cleaning']}s.")

            # ── FASE 3: Feature Engineering ──
            self.current_phase = "engineering"
            logger.info("=" * 70)
            logger.info("FASE 3: FEATURE ENGINEERING")
            logger.info("=" * 70)
            t0 = time.time()

            self.df_featured = await loop.run_in_executor(
                None, run_feature_engineering, df_clean
            )

            self.phase_durations["engineering"] = round(time.time() - t0, 2)
            logger.info(f"Fase 3 completada en {self.phase_durations['engineering']}s.")

            # ── FASE 4: Balanceo ──
            self.current_phase = "balancing"
            logger.info("=" * 70)
            logger.info("FASE 4: BALANCEO DE CLASES")
            logger.info("=" * 70)
            t0 = time.time()

            self.balanced_data = await loop.run_in_executor(
                None, prepare_balanced_dataset, self.df_featured
            )

            self.phase_durations["balancing"] = round(time.time() - t0, 2)
            logger.info(f"Fase 4 completada en {self.phase_durations['balancing']}s.")

            # ── FASE 5: Entrenamiento de Agentes (3 oleadas) ──
            self.current_phase = "training"
            logger.info("=" * 70)
            logger.info("FASE 5: ENTRENAMIENTO DE AGENTES")
            logger.info("=" * 70)
            t0 = time.time()

            await self._train_agents_in_waves()

            self.phase_durations["training"] = round(time.time() - t0, 2)
            logger.info(f"Fase 5 completada en {self.phase_durations['training']}s.")

            # ── Finalización ──
            self.status = "ready"
            self.current_phase = "completed"
            self.pipeline_duration = round(time.time() - self.pipeline_start, 2)

            logger.info("=" * 70)
            logger.info(f"PIPELINE COMPLETO — Duración total: {self.pipeline_duration}s")
            logger.info("=" * 70)

            return self.get_status()

        except Exception as e:
            self.status = "error"
            self.current_phase = f"error: {str(e)}"
            self.pipeline_duration = round(time.time() - self.pipeline_start, 2)
            logger.error(f"Error en pipeline: {e}", exc_info=True)
            raise

    async def _train_agents_in_waves(self):
        """
        Entrena los 9 agentes en 3 oleadas paralelas respetando dependencias:
        - Oleada 1: A1 (Credit Scoring), A5 (Clustering), A6 (Producto), A7 (Edad)
        - Oleada 2: A2 (Alerta), A3 (Roll-Rate) — necesitan A7 (filtro menores)
        - Oleada 3: A4 (Sobreend.), A8 (Fechas), A9 (Familiar) — necesitan A1
        """
        df = self.df_featured
        balanced = self.balanced_data

        # ──────────────── OLEADA 1: Independientes ────────────────
        logger.info("▶ Oleada 1: Agentes independientes (A1, A5, A6, A7)")

        wave_1_tasks = [
            self.agents["credit_scoring"].train(
                df,
                X_train=balanced["X_train"],
                X_test=balanced["X_test"],
                y_train=balanced["y_train"],
                y_test=balanced["y_test"],
                feature_columns=balanced["feature_columns"],
            ),
            self.agents["collection_segments"].train(df),
            self.agents["product_risk"].train(df),
            self.agents["age_validation"].train(df),
        ]

        wave_1_results = await asyncio.gather(*wave_1_tasks, return_exceptions=True)
        for i, result in enumerate(wave_1_results):
            if isinstance(result, Exception):
                logger.error(f"Error en oleada 1, tarea {i}: {result}")

        # ──────────────── OLEADA 2: Dependen de A7 ────────────────
        logger.info("▶ Oleada 2: Post-validación de edad (A2, A3)")

        # Filtrar menores si A7 fue exitoso
        minor_ids = []
        if self.agents["age_validation"].status == "ready":
            minor_ids = self.agents["age_validation"].results.get("minor_client_ids", [])

        if minor_ids and "v_ah_cliente" in df.columns:
            df_adults = df[~df["v_ah_cliente"].isin(minor_ids)].copy()
            logger.info(f"Filtrados {len(minor_ids)} menores. Adultos: {len(df_adults)}.")
        else:
            df_adults = df

        wave_2_tasks = [
            self.agents["early_warning"].train(df_adults),
            self.agents["roll_rate"].train(df_adults),
        ]

        wave_2_results = await asyncio.gather(*wave_2_tasks, return_exceptions=True)
        for i, result in enumerate(wave_2_results):
            if isinstance(result, Exception):
                logger.error(f"Error en oleada 2, tarea {i}: {result}")

        # ──────────────── OLEADA 3: Dependen de A1 ────────────────
        logger.info("▶ Oleada 3: Post-credit scoring (A4, A8, A9)")

        wave_3_tasks = [
            self.agents["over_indebtedness"].train(df, credit_scores=self.agents["credit_scoring"]),
            self.agents["date_optimization"].train(df),
            self.agents["family_impact"].train(df, credit_scores=self.agents["credit_scoring"]),
        ]

        wave_3_results = await asyncio.gather(*wave_3_tasks, return_exceptions=True)
        for i, result in enumerate(wave_3_results):
            if isinstance(result, Exception):
                logger.error(f"Error en oleada 3, tarea {i}: {result}")

    def get_dashboard_summary(self) -> dict:
        """Genera resumen consolidado de todos los agentes para el Dashboard."""
        summary = {
            "pipeline_status": self.status,
            "pipeline_duration_seconds": self.pipeline_duration,
            "phase_durations": self.phase_durations,
            "total_agents": len(self.agents),
            "agents_ready": sum(1 for a in self.agents.values() if a.status == "ready"),
            "agents_error": sum(1 for a in self.agents.values() if a.status == "error"),
            "agents": {},
        }

        for agent_id, agent in self.agents.items():
            summary["agents"][agent_id] = agent.get_summary()

        return summary

    def get_socio_profile(self, socio_id) -> dict:
        """Genera perfil 360° de un socio con datos de todos los agentes."""
        profile = {
            "v_ah_cliente": socio_id,
            "agents": {},
        }

        # Buscar datos del socio en el DataFrame featured
        socio_data = {}
        if self.df_featured is not None and "v_ah_cliente" in self.df_featured.columns:
            match = self.df_featured[self.df_featured["v_ah_cliente"] == socio_id]
            if not match.empty:
                # Tomar último período
                if "periodo" in match.columns:
                    match = match.sort_values("periodo", ascending=False).head(1)
                socio_data = match.iloc[0].to_dict()

        profile["socio_data"] = {
            k: (v if not pd.isna(v) else None) if not isinstance(v, (pd.Timestamp,)) else str(v)
            for k, v in socio_data.items()
            if k not in ["v_ah_nombre"]  # Excluir nombre por privacidad
        }

        return profile

    def get_client_data(self, cliente_id: float) -> dict:
        """Busca y retorna el dict de datos de un cliente en el dataframe df_featured."""
        if self.df_featured is None or self.df_featured.empty:
            raise ValueError("No hay datos cargados en el pipeline.")

        # Buscar cliente
        match = self.df_featured[self.df_featured["v_ah_cliente"] == cliente_id]
        if match.empty:
            raise ValueError(f"Cliente con ID {cliente_id} no encontrado.")

        # Tomar el ultimo registro del cliente
        if "periodo" in match.columns:
            match = match.sort_values("periodo", ascending=False)
        row = match.iloc[0]

        # Convertir a dict nativo de python
        d = row.to_dict()
        for k, v in d.items():
            if isinstance(v, (np.integer, np.floating)):
                d[k] = v.item() if hasattr(v, "item") else float(v)
            elif pd.isna(v):
                d[k] = None
        return d

    async def run_client_scoring(self, client_data: dict) -> dict:
        """
        Ejecuta las predicciones de los 9 agentes y consolida la salida.
        """
        if self.status != "ready":
            raise ValueError("El pipeline no esta listo. Ejecute el entrenamiento primero.")

        # Ejecutar en paralelo todos los agentes
        tasks = {
            agent_id: agent.predict(client_data)
            for agent_id, agent in self.agents.items()
        }

        results = {}
        for agent_id, task in tasks.items():
            try:
                results[agent_id] = await task
            except Exception as e:
                logger.error(f"Error prediciendo con agente {agent_id}: {e}")
                results[agent_id] = {"error": str(e), "status": "error"}

        # Agrupar resultados
        # 1. Age Validation
        age_res = results.get("age_validation", {})
        is_minor = age_res.get("is_minor", False)
        is_blocked_age = age_res.get("is_blocked", False)

        # 2. Credit Scoring
        cs_res = results.get("credit_scoring", {})
        cs_score = cs_res.get("score", 0)
        prob_default = cs_res.get("probability_default", 0)

        # 3. Early Warning
        ew_res = results.get("early_warning", {})
        ew_score = ew_res.get("alert_score", 0)
        ew_level = ew_res.get("alert_level", "Normal")

        # 4. Roll Rate
        rr_res = results.get("roll_rate", {})
        rr_prob = rr_res.get("prob_deterioro", 0)

        # 5. Over Indebtedness
        oi_res = results.get("over_indebtedness", {})
        is_overindebted = oi_res.get("is_overindebted", False)
        oi_score = oi_res.get("overindebtedness_score", 0)

        # 6. Collection Segments
        col_res = results.get("collection_segments", {})
        cluster_name = col_res.get("cluster_name", "Recuperable facil")
        strategy = col_res.get("strategy", "")

        # 7. Product Risk
        pr_res = results.get("product_risk", {})
        prod_risk_score = pr_res.get("product_risk", {}).get("risk_score", 0)

        # 8. Date Optimization
        do_res = results.get("date_optimization", {})
        fecha_sugerida = do_res.get("fecha_sugerida_cobro", 15)

        # 9. Family Impact
        fi_res = results.get("family_impact", {})
        ingreso_ajustado = fi_res.get("ingreso_ajustado", 0)
        vulnerability_score = fi_res.get("vulnerability_score", 0)

        # Elegibilidad consolidada
        elegible_edad = not is_blocked_age
        elegible_riesgo = cs_score >= 350
        elegible_capacidad = not is_overindebted
        recursos_ok = ingreso_ajustado >= 0
        elegibilidad = elegible_edad and elegible_riesgo and elegible_capacidad and recursos_ok

        # Riesgo Global
        risk_scores = [
            prob_default * 100,
            ew_score * 100,
            rr_prob * 100,
            prod_risk_score
        ]
        riesgo_global = round(sum(risk_scores) / len(risk_scores), 2)

        # Canal de Cobranza sugerido
        mapping = {
            "Recuperable facil": "whatsapp",
            "Negociacion": "llamada",
            "Pre-juridico": "llamada",
            "Castigo potencial": "legal"
        }
        canal_cobranza = mapping.get(cluster_name, "llamada")

        # Alertas activas
        alertas = []
        if is_minor:
            alertas.append("MENOR_EDAD")
        if "Normal" not in str(ew_level):
            alertas.append(str(ew_level).replace("\U0001f534 ", "").replace("\U0001f7e0 ", "").replace("\U0001f7e1 ", "").upper())
        if is_overindebted:
            alertas.append("SOBREENDEUDAMIENTO")
        if rr_prob > 0.5:
            alertas.append("DETERIORO")
        if vulnerability_score > 60:
            alertas.append("CARGA_FAMILIAR_ALTA")

        # Acciones priorizadas
        acciones = []
        if is_minor:
            acciones.append("Bloquear operaciones por minoria de edad.")
        if is_overindebted:
            acciones.append("Socio sobreendeudado. Evitar incremento de cupo o nuevas operaciones.")
        if "Normal" not in str(ew_level):
            acciones.append("Alerta temprana de mora. Realizar gestion preventiva de cobro.")
        if rr_prob > 0.5:
            acciones.append("Alto riesgo de migracion a peor estado de mora. Realizar contacto telefonico.")
        if do_res.get("fecha_sugerida_cobro") is not None:
            acciones.append(f"Mover fecha de cobro sugerida al dia {fecha_sugerida}.")
        if strategy:
            acciones.append(f"Estrategia de cobranza: {strategy}")

        # Bloqueos
        bloqueos = []
        if is_minor:
            bloqueos.append("age_validation")
        if oi_score >= 80:
            bloqueos.append("over_indebtedness")
        if cs_score < 350:
            bloqueos.append("credit_scoring")

        return {
            "cliente_id": client_data.get("v_ah_cliente") or client_data.get("cliente_id"),
            "riesgo_global": riesgo_global,
            "elegibilidad_credito": elegibilidad,
            "canal_cobranza": canal_cobranza,
            "dia_pago_sugerido": int(fecha_sugerida),
            "alertas_activas": alertas,
            "acciones_priorizadas": acciones[:8],
            "bloqueos": bloqueos,
            "agentes": results
        }


# Instancia global del engine (singleton)
pipeline_engine = PipelineEngine()
