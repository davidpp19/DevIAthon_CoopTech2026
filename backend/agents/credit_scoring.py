"""
CoopTech Backend — Agente 1: Credit Scoring.

Ensemble de Random Forest + Regresión Logística con Voting suave
para generar un score de riesgo crediticio de 0 a 1000.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from agents.base_agent import BaseAgent
from config import CREDIT_SCORING, MODELS_DIR

logger = logging.getLogger(__name__)


# Mapeo de score numérico a calificación letra
def _score_to_grade(score: float) -> str:
    """Convierte score 0-1000 a calificación A-E."""
    if score >= 800:
        return "A"
    elif score >= 650:
        return "B"
    elif score >= 500:
        return "C"
    elif score >= 350:
        return "D"
    else:
        return "E"


class CreditScoringAgent(BaseAgent):
    """
    Modelo de Credit Scoring basado en ensemble (RF + LogReg).
    Genera un score de 0-1000 y una calificación A/B/C/D/E.
    """

    def __init__(self):
        super().__init__(
            agent_id="credit_scoring",
            agent_name="Modelo de Credit Scoring",
        )
        self.model: Optional[VotingClassifier] = None
        self.feature_columns: list[str] = []
        self.feature_importances: dict[str, float] = {}

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """
        Entrena el ensemble RF + LogReg.

        Espera recibir en kwargs:
            X_train, X_test, y_train, y_test, feature_columns
        """
        self._set_training()

        try:
            X_train = kwargs["X_train"]
            X_test = kwargs["X_test"]
            y_train = kwargs["y_train"]
            y_test = kwargs["y_test"]
            self.feature_columns = kwargs.get("feature_columns", list(X_train.columns))

            # Construir ensemble
            rf = RandomForestClassifier(
                n_estimators=CREDIT_SCORING["rf_n_estimators"],
                max_depth=CREDIT_SCORING["rf_max_depth"],
                min_samples_split=CREDIT_SCORING["rf_min_samples_split"],
                random_state=CREDIT_SCORING["random_state"],
                n_jobs=-1,
            )

            lr = LogisticRegression(
                C=CREDIT_SCORING["lr_C"],
                max_iter=CREDIT_SCORING["lr_max_iter"],
                random_state=CREDIT_SCORING["random_state"],
                solver="lbfgs",
            )

            self.model = VotingClassifier(
                estimators=[("rf", rf), ("lr", lr)],
                voting="soft",
            )

            # Entrenar en un thread para no bloquear el event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.model.fit, X_train, y_train)

            # Evaluar
            y_pred = await loop.run_in_executor(None, self.model.predict, X_test)
            y_proba = await loop.run_in_executor(
                None, self.model.predict_proba, X_test
            )

            # Métricas
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_test, y_proba[:, 1])),
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True, zero_division=0
                ),
            }

            # Feature importance (del Random Forest)
            rf_model = self.model.named_estimators_["rf"]
            importances = rf_model.feature_importances_
            self.feature_importances = dict(
                sorted(
                    zip(self.feature_columns, importances.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

            # Generar scores para todo el test set
            scores = (y_proba[:, 0] * 1000).round(0).astype(int)  # P(no riesgo) * 1000
            grades = [_score_to_grade(s) for s in scores]

            # Distribución de calificaciones
            grade_dist = pd.Series(grades).value_counts().to_dict()

            results = {
                "scores_test": scores.tolist(),
                "grades_test": grades,
                "grade_distribution": grade_dist,
                "top_features": dict(list(self.feature_importances.items())[:10]),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }

            # Guardar modelo
            model_path = MODELS_DIR / "credit_scoring.joblib"
            joblib.dump(self.model, model_path)
            logger.info(f"Modelo guardado en {model_path}")

            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    async def predict(self, input_data: dict) -> dict:
        """
        Predicción individual: genera score y calificación para un socio.
        """
        if self.model is None:
            raise RuntimeError("Modelo no entrenado. Ejecute train() primero.")

        # Construir vector de features
        features = pd.DataFrame([input_data])[self.feature_columns]
        features = features.fillna(0).replace([np.inf, -np.inf], 0)

        proba = self.model.predict_proba(features)[0]
        score = int(round(proba[0] * 1000))  # P(no riesgo) * 1000
        grade = _score_to_grade(score)
        prob_default = float(proba[1])

        return {
            "agent_id": self.agent_id,
            "score": score,
            "grade": grade,
            "probability_default": round(prob_default, 4),
            "risk_level": "alto" if score < 350 else "medio" if score < 650 else "bajo",
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "metrics": self.metrics,
            "grade_distribution": self.results.get("grade_distribution", {}),
            "top_features": self.results.get("top_features", {}),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
