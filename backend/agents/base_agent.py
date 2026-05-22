"""
CoopTech Backend — Clase Abstracta BaseAgent.

Define la interfaz común que todos los 9 agentes analíticos deben implementar.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd


class BaseAgent(ABC):
    """
    Interfaz base para todos los agentes analíticos del sistema CoopTech.

    Cada agente encapsula un modelo de ML o un sistema basado en reglas,
    con métodos estandarizados de entrenamiento, predicción y resumen.
    """

    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.status: Literal["idle", "training", "ready", "error"] = "idle"
        self.trained_at: Optional[datetime] = None
        self.training_duration_seconds: float = 0.0
        self.metrics: dict[str, Any] = {}
        self.results: dict[str, Any] = {}
        self.logger = logging.getLogger(f"agent.{agent_id}")

    @abstractmethod
    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        Entrena el modelo con el DataFrame procesado.

        Args:
            df: DataFrame con features ya derivados.
            **kwargs: Parámetros adicionales específicos del agente.

        Returns:
            dict con métricas de entrenamiento y resultados.
        """
        pass

    @abstractmethod
    async def predict(self, input_data: dict) -> dict:
        """
        Predicción individual para un socio.

        Args:
            input_data: dict con los features del socio.

        Returns:
            dict con la predicción y metadatos.
        """
        pass

    @abstractmethod
    def get_summary(self) -> dict:
        """
        Resumen de métricas y resultados para el Dashboard.

        Returns:
            dict con resumen ejecutivo del agente.
        """
        pass

    def get_status(self) -> dict:
        """Retorna el estado actual del agente."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_duration_seconds": self.training_duration_seconds,
        }

    def _set_training(self):
        """Marca el inicio del entrenamiento."""
        self.status = "training"
        self._train_start = datetime.now()
        self.logger.info(f"[{self.agent_id}] Entrenamiento iniciado.")

    def _set_ready(self, metrics: dict, results: dict):
        """Marca la finalización exitosa del entrenamiento."""
        self.status = "ready"
        self.trained_at = datetime.now()
        self.training_duration_seconds = (
            datetime.now() - self._train_start
        ).total_seconds()
        self.metrics = metrics
        self.results = results
        self.logger.info(
            f"[{self.agent_id}] Entrenamiento completado en "
            f"{self.training_duration_seconds:.2f}s."
        )

    def _set_error(self, error: Exception):
        """Marca un error en el agente."""
        self.status = "error"
        self.logger.error(f"[{self.agent_id}] Error: {error}")
