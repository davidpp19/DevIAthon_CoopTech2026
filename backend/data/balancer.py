"""
CoopTech Backend — Módulo de Balanceo de Clases.

SMOTE para sobremuestreo sintético de la clase minoritaria.
Split estratificado train/test ANTES de balancear (evitar data leakage).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import BALANCER, CREDIT_SCORING

logger = logging.getLogger(__name__)

# Intentar importar SMOTE, con fallback a oversampling aleatorio
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning(
        "imbalanced-learn no disponible. Se usará oversampling aleatorio como fallback."
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna las columnas de features para modelado.
    Excluye columnas de ID, nombre, fecha, target y metadatos.
    """
    exclude_patterns = [
        "v_ah_cliente", "v_ah_cuenta", "v_ah_nombre", "cliente",
        "fecha_", "v_fecha_", "periodo", "tipo_cuenta", "nacionalidad",
        "estado_cta", "sexo", "estado_civil", "fuente_alerta",
        "riesgo_alto",  # target
    ]

    feature_cols = []
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64, float, int]:
            skip = False
            for pattern in exclude_patterns:
                if pattern in col:
                    skip = True
                    break
            if not skip:
                feature_cols.append(col)

    return feature_cols


def split_train_test(
    df: pd.DataFrame,
    target_col: str = "riesgo_alto",
    test_size: Optional[float] = None,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split estratificado train/test.
    SE HACE ANTES del balanceo para evitar data leakage.

    Returns:
        X_train, X_test, y_train, y_test
    """
    test_size = test_size or CREDIT_SCORING["test_size"]
    random_state = random_state or CREDIT_SCORING["random_state"]

    if target_col not in df.columns:
        raise ValueError(f"Columna target '{target_col}' no encontrada en DataFrame.")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Reemplazar inf y NaN residuales
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        f"Split train/test: train={X_train.shape[0]}, test={X_test.shape[0]}, "
        f"ratio_positivos_train={y_train.mean():.3f}, "
        f"ratio_positivos_test={y_test.mean():.3f}"
    )

    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_neighbors: Optional[int] = None,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Aplica SMOTE al training set para balancear clases.
    Fallback a oversampling aleatorio si imblearn no está disponible.

    Args:
        X_train: Features del training set.
        y_train: Target del training set.

    Returns:
        X_resampled, y_resampled (balanceados).
    """
    k_neighbors = k_neighbors or BALANCER["smote_k_neighbors"]
    random_state = random_state or BALANCER["random_state"]

    before_counts = y_train.value_counts().to_dict()
    logger.info(f"Antes del balanceo: {before_counts}")

    # Verificar que hay suficientes muestras para SMOTE
    minority_count = y_train.value_counts().min()
    if minority_count < k_neighbors + 1:
        k_neighbors = max(1, minority_count - 1)
        logger.warning(
            f"Reduciendo k_neighbors a {k_neighbors} (muestras minoritarias: {minority_count})."
        )

    if HAS_IMBLEARN and minority_count >= 2:
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        y_res = pd.Series(y_res, name=y_train.name)
        method = "SMOTE"
    else:
        # Fallback: oversampling aleatorio de la clase minoritaria
        X_res, y_res = _random_oversample(X_train, y_train, random_state)
        method = "Random Oversampling (fallback)"

    after_counts = y_res.value_counts().to_dict()
    logger.info(f"Después del balanceo ({method}): {after_counts}")

    return X_res, y_res


def _random_oversample(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversampling aleatorio de la clase minoritaria."""
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()
    n_majority = (y == majority_class).sum()

    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]

    # Duplicar la clase minoritaria hasta igualar la mayoritaria
    n_to_sample = n_majority - len(X_minority)
    if n_to_sample > 0:
        X_upsampled = X_minority.sample(n=n_to_sample, replace=True, random_state=random_state)
        y_upsampled = y_minority.sample(n=n_to_sample, replace=True, random_state=random_state)

        X_res = pd.concat([X, X_upsampled], ignore_index=True)
        y_res = pd.concat([y, y_upsampled], ignore_index=True)
    else:
        X_res = X.copy()
        y_res = y.copy()

    return X_res, y_res


def prepare_balanced_dataset(
    df: pd.DataFrame,
    target_col: str = "riesgo_alto",
) -> dict:
    """
    Pipeline completo: split → balanceo.
    Retorna un dict con todos los datasets necesarios para entrenamiento.

    Returns:
        {
            "X_train": DataFrame balanceado,
            "X_test": DataFrame original (no balanceado),
            "y_train": Series balanceada,
            "y_test": Series original,
            "feature_columns": list de nombres de features,
            "df_full": DataFrame completo con features (sin split),
        }
    """
    logger.info("=" * 60)
    logger.info("INICIO DE BALANCEO DE CLASES")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = split_train_test(df, target_col)
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)

    feature_cols = get_feature_columns(df)

    return {
        "X_train": X_train_balanced,
        "X_test": X_test,
        "y_train": y_train_balanced,
        "y_test": y_test,
        "feature_columns": feature_cols,
        "df_full": df,
    }
