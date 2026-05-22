"""
CoopTech Backend — Agente 5: Segmentación para Estrategias de Cobranza.

Clustering K-Means para agrupar socios en perfiles de cobranza,
con selección automática de k óptimo vía Silhouette Score.
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from agents.base_agent import BaseAgent
from config import CLUSTERING

logger = logging.getLogger(__name__)

# Estrategias sugeridas por cluster
COLLECTION_STRATEGIES = {
    "Recuperable fácil": "Recordatorio SMS/email + llamada amable. Alta probabilidad de pago voluntario.",
    "Negociación": "Contacto personalizado + oferta de reestructuración. Socio con capacidad pero requiere flexibilidad.",
    "Pre-jurídico": "Notificación formal + visita domiciliaria. Riesgo medio-alto, requiere acción inmediata.",
    "Castigo potencial": "Escalamiento a jurídico. Análisis de garantías. Provisión completa recomendada.",
}


class CollectionSegmentsAgent(BaseAgent):
    """
    Segmentación K-Means para estrategias diferenciadas de cobranza.
    """

    def __init__(self):
        super().__init__(
            agent_id="collection_segments",
            agent_name="Segmentación para Cobranza",
        )
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_profiles: dict = {}
        self.cluster_data: pd.DataFrame = pd.DataFrame()
        self.optimal_k: int = 4

    async def train(self, df: pd.DataFrame, **kwargs) -> dict:
        """Entrena K-Means con k óptimo y genera perfiles de cluster."""
        self._set_training()

        try:
            loop = asyncio.get_event_loop()
            metrics, results = await loop.run_in_executor(
                None, self._train_kmeans, df
            )
            self._set_ready(metrics, results)
            return {"metrics": metrics, "results": results}

        except Exception as e:
            self._set_error(e)
            raise

    def _get_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selecciona y prepara features para clustering."""
        feature_cols = []
        candidates = [
            "saldo_efectivo", "dias_sin_movimiento", "flujo_neto",
            "antiguedad_cuenta_dias", "productos_digitales",
            "ratio_ingreso_egreso", "volumen_total", "ratio_bloqueo",
        ]
        for col in candidates:
            if col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            raise ValueError("No se encontraron features para clustering.")

        features = df[feature_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        return features

    def _train_kmeans(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """Entrena K-Means, selecciona k óptimo, genera perfiles."""
        # Tomar último período si hay múltiples
        if "periodo" in df.columns:
            last_periodo = df["periodo"].max()
            df_work = df[df["periodo"] == last_periodo].copy()
        else:
            df_work = df.copy()

        features = self._get_clustering_features(df_work)
        feature_cols = features.columns.tolist()

        # Escalar features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features)

        # Buscar k óptimo
        k_min, k_max = CLUSTERING["k_range"]
        silhouette_scores = {}

        for k in range(k_min, k_max + 1):
            km = KMeans(
                n_clusters=k,
                random_state=CLUSTERING["random_state"],
                max_iter=CLUSTERING["max_iter"],
                n_init=CLUSTERING["n_init"],
            )
            labels = km.fit_predict(X_scaled)
            if len(set(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                silhouette_scores[k] = round(sil, 4)

        # Seleccionar k con mejor silhouette
        if silhouette_scores:
            self.optimal_k = max(silhouette_scores, key=silhouette_scores.get)
        else:
            self.optimal_k = 4

        # Entrenar modelo final con k óptimo
        self.model = KMeans(
            n_clusters=self.optimal_k,
            random_state=CLUSTERING["random_state"],
            max_iter=CLUSTERING["max_iter"],
            n_init=CLUSTERING["n_init"],
        )
        labels = self.model.fit_predict(X_scaled)

        # Asignar nombres descriptivos a clusters basados en centroides
        cluster_names = self._assign_cluster_names(
            pd.DataFrame(X_scaled, columns=feature_cols), labels
        )

        # Generar perfiles
        df_work = df_work.copy()
        df_work["cluster_id"] = labels
        df_work["cluster_name"] = df_work["cluster_id"].map(cluster_names)

        self.cluster_profiles = {}
        for cid in range(self.optimal_k):
            cluster_df = df_work[df_work["cluster_id"] == cid]
            name = cluster_names.get(cid, f"Cluster {cid}")
            self.cluster_profiles[name] = {
                "n_socios": len(cluster_df),
                "pct_total": round(len(cluster_df) / len(df_work) * 100, 1),
                "avg_saldo": round(float(cluster_df["saldo_disponible"].mean()), 2) if "saldo_disponible" in cluster_df.columns else 0,
                "avg_ingresos": round(float(cluster_df["ingresos"].mean()), 2) if "ingresos" in cluster_df.columns else 0,
                "avg_dias_inactivo": round(float(cluster_df["dias_sin_movimiento"].mean()), 1) if "dias_sin_movimiento" in cluster_df.columns else 0,
                "strategy": COLLECTION_STRATEGIES.get(name, "Análisis individual requerido."),
            }

        self.cluster_data = df_work[["v_ah_cliente", "cluster_id", "cluster_name"]].copy() if "v_ah_cliente" in df_work.columns else pd.DataFrame()

        metrics = {
            "optimal_k": self.optimal_k,
            "silhouette_scores": silhouette_scores,
            "best_silhouette": silhouette_scores.get(self.optimal_k, 0),
            "inertia": float(self.model.inertia_),
            "n_socios_segmentados": len(df_work),
        }

        results = {
            "cluster_profiles": self.cluster_profiles,
            "cluster_distribution": {name: p["n_socios"] for name, p in self.cluster_profiles.items()},
            "feature_columns": feature_cols,
        }

        return metrics, results

    def _assign_cluster_names(self, X: pd.DataFrame, labels: np.ndarray) -> dict:
        """Asigna nombres descriptivos basados en centroides."""
        strategy_names = list(COLLECTION_STRATEGIES.keys())
        centroids_df = pd.DataFrame(X)
        centroids_df["label"] = labels

        # Ranking por score compuesto (menor saldo + mayor inactividad = peor)
        cluster_scores = {}
        for cid in range(self.optimal_k):
            cluster = centroids_df[centroids_df["label"] == cid]
            # Score: alto = peor situación
            if len(cluster) > 0:
                score = -cluster.iloc[:, 0].mean() + cluster.iloc[:, 1].mean() if X.shape[1] >= 2 else 0
                cluster_scores[cid] = score

        # Ordenar clusters de mejor a peor
        sorted_clusters = sorted(cluster_scores, key=cluster_scores.get)

        names = {}
        for i, cid in enumerate(sorted_clusters):
            if i < len(strategy_names):
                names[cid] = strategy_names[i]
            else:
                names[cid] = f"Segmento {i + 1}"

        return names

    async def predict(self, input_data: dict) -> dict:
        """Asigna un cluster a un socio individual."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Agente no entrenado.")

        features = self._get_clustering_features(pd.DataFrame([input_data]))
        X_scaled = self.scaler.transform(features)
        cluster_id = int(self.model.predict(X_scaled)[0])

        cluster_names = {v: k for k, v in enumerate(self.cluster_profiles.keys())}
        name_map = {v: k for k, v in cluster_names.items()}
        cluster_name = name_map.get(cluster_id, f"Cluster {cluster_id}")

        return {
            "agent_id": self.agent_id,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "strategy": self.cluster_profiles.get(cluster_name, {}).get("strategy", ""),
        }

    def get_summary(self) -> dict:
        """Resumen para el Dashboard."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "optimal_k": self.optimal_k,
            "metrics": self.metrics,
            "cluster_profiles": self.cluster_profiles,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
        }
