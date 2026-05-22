"""
CoopTech Backend — Router del Dashboard.

Endpoints consolidados para alimentar el Dashboard frontend.
Persiste resultados en Supabase y lee de ahí como fuente primaria.
"""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from orchestrator.engine import pipeline_engine
from schemas.schemas import (
    DashboardSummary,
    KPIResponse,
    SocioProfile,
    DashboardSegmentos,
    SegmentoProducto,
)
import supabase_client as sb

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def get_dashboard_summary():
    """
    Retorna el resumen consolidado de los 9 agentes para el Dashboard.
    Incluye métricas clave, distribuciones y estados.
    """
    summary = pipeline_engine.get_dashboard_summary()
    return DashboardSummary(**summary)


@router.get("/socios/{socio_id}", response_model=SocioProfile)
async def get_socio_profile(socio_id: float):
    """
    Retorna el perfil 360° de un socio específico.
    Incluye datos de todos los agentes disponibles.
    """
    if pipeline_engine.status != "ready":
        raise HTTPException(
            status_code=400,
            detail="Pipeline no está listo. Ejecute POST /api/v1/pipeline/run primero.",
        )

    profile = pipeline_engine.get_socio_profile(socio_id)

    # Ejecutar predicciones en todos los agentes disponibles
    for agent_id, agent in pipeline_engine.agents.items():
        if agent.status == "ready":
            try:
                socio_data = profile.get("socio_data", {})
                prediction = await agent.predict(socio_data)
                profile["agents"][agent_id] = prediction
            except Exception as e:
                profile["agents"][agent_id] = {
                    "error": str(e),
                    "status": "prediction_failed",
                }

    return SocioProfile(**profile)


@router.get("/kpis", response_model=KPIResponse)
async def get_kpis():
    """
    Retorna KPIs globales consolidados para el Dashboard.
    Intenta leer de Supabase primero; si no hay datos, calcula del pipeline.
    """
    kpis = KPIResponse(pipeline_status=pipeline_engine.status)

    if pipeline_engine.status != "ready":
        # Intentar leer de Supabase como fallback
        cached = await sb.get_kpis()
        if cached:
            cached.pop("id", None)
            cached.pop("updated_at", None)
            try:
                return KPIResponse(**cached)
            except Exception:
                pass
        return kpis

    # Extraer KPIs de cada agente y del dataframe
    agents = pipeline_engine.agents
    df = pipeline_engine.df_featured

    if df is not None and not df.empty:
        # Tomar el ultimo periodo para fotos estaticas
        if "periodo" in df.columns:
            df_last = df[df["periodo"] == df["periodo"].max()]
        else:
            df_last = df

        kpis.total_socios = int(df_last["v_ah_cliente"].nunique())
        kpis.tasa_mora_pct = round(
            float((df_last["dias_sin_movimiento"] > 30).mean() * 100), 2
        )
        kpis.alertas_desvio = int(
            (df_last["fuente_alerta"] != "sin_alerta").sum()
        )
        kpis.menores_detectados = int((df_last["menor_edad"] == 1).sum())

        # Distribucion proxy de mora por buckets
        kpis.distribucion_riesgo = {
            "sin_mora": int((df_last["dias_sin_movimiento"] <= 30).sum()),
            "mora_30d": int(
                (
                    (df_last["dias_sin_movimiento"] > 30)
                    & (df_last["dias_sin_movimiento"] <= 60)
                ).sum()
            ),
            "mora_60d": int(
                (
                    (df_last["dias_sin_movimiento"] > 60)
                    & (df_last["dias_sin_movimiento"] <= 90)
                ).sum()
            ),
            "mora_90d": int((df_last["dias_sin_movimiento"] > 90).sum()),
        }

    # Total socios / menores bloqueados de la validacion de edad
    if agents["age_validation"].status == "ready":
        kpis.menores_bloqueados = agents["age_validation"].metrics.get(
            "menores_bloqueados", 0
        )

    # Tasa de riesgo alto del credit scoring
    if agents["credit_scoring"].status == "ready":
        grade_dist = agents["credit_scoring"].results.get("grade_distribution", {})
        total_scores = sum(grade_dist.values()) if grade_dist else 1
        high_risk = grade_dist.get("D", 0) + grade_dist.get("E", 0)
        kpis.tasa_riesgo_alto = (
            round(high_risk / total_scores * 100, 2) if total_scores > 0 else 0
        )

    # Socios en alerta y distribucion de alertas
    if agents["early_warning"].status == "ready":
        alert_dist = agents["early_warning"].metrics.get("alert_distribution", {})
        kpis.socios_en_alerta = sum(
            v for k, v in alert_dist.items() if "Normal" not in k
        )
        kpis.distribucion_alerta = {
            "normal": alert_dist.get("🟢 Normal", 0),
            "alerta_30d": alert_dist.get("🟡 Alerta 30d", 0),
            "alerta_60d": alert_dist.get("🟠 Alerta 60d", 0),
            "alerta_90d": alert_dist.get("🔴 Alerta 90d", 0),
        }

    # Sobreendeudamiento
    if agents["over_indebtedness"].status == "ready":
        kpis.socios_sobreendeudados = agents["over_indebtedness"].metrics.get(
            "socios_sobreendeudados", 0
        )

    # Clusters y distribucion por canal
    if agents["collection_segments"].status == "ready":
        kpis.clusters_count = agents["collection_segments"].optimal_k
        profiles = agents["collection_segments"].cluster_profiles
        kpis.distribucion_canal = {
            "whatsapp": profiles.get("Recuperable fácil", {}).get("n_socios", 0),
            "llamada": profiles.get("Negociación", {}).get("n_socios", 0)
            + profiles.get("Pre-jurídico", {}).get("n_socios", 0),
            "legal": profiles.get("Castigo potencial", {}).get("n_socios", 0),
        }

    # Persistir a Supabase en background
    asyncio.create_task(_persist_kpis(kpis.model_dump()))

    return kpis


async def _persist_kpis(kpis_dict: dict):
    """Persiste KPIs a Supabase en background sin bloquear la respuesta."""
    try:
        await sb.upload_kpis(kpis_dict)
    except Exception as e:
        logger.warning("Error persistiendo KPIs a Supabase: %s", e)


@router.get("/segmentos", response_model=DashboardSegmentos)
async def get_dashboard_segmentos():
    """
    Retorna la lista de segmentos por producto y su distribucion por canal de cobranza.
    """
    agents = pipeline_engine.agents

    por_producto = []
    if agents["product_risk"].status == "ready":
        for prod, info in agents["product_risk"].risk_by_product.items():
            por_producto.append(
                SegmentoProducto(
                    prod_bancario=str(prod),
                    n_socios=info["n_socios"],
                    tasa_mora_pct=info["tasa_riesgo"],
                    score_riesgo=info["risk_score"] / 100.0,
                )
            )

    por_canal = {}
    if agents["collection_segments"].status == "ready":
        profiles = agents["collection_segments"].cluster_profiles
        por_canal = {
            "whatsapp": profiles.get("Recuperable fácil", {}).get("n_socios", 0),
            "llamada": profiles.get("Negociación", {}).get("n_socios", 0)
            + profiles.get("Pre-jurídico", {}).get("n_socios", 0),
            "legal": profiles.get("Castigo potencial", {}).get("n_socios", 0),
        }

    result = DashboardSegmentos(por_producto=por_producto, por_canal_cobranza=por_canal)

    # Persistir segmentos a Supabase en background
    asyncio.create_task(
        _persist_segmentos([s.model_dump() for s in por_producto])
    )

    return result


async def _persist_segmentos(segmentos: list[dict]):
    """Persiste segmentos a Supabase en background."""
    try:
        await sb.upload_segmentos(segmentos)
    except Exception as e:
        logger.warning("Error persistiendo segmentos a Supabase: %s", e)


@router.get("/clusters")
async def get_dashboard_clusters():
    """
    Retorna el conteo de socios en cada cluster para graficar.
    """
    agents = pipeline_engine.agents
    clusters = {}
    if agents["collection_segments"].status == "ready":
        dist = agents["collection_segments"].results.get("cluster_distribution", {})
        clusters = {f"cluster_{name}": size for name, size in dist.items()}

    # Persistir a Supabase en background
    if clusters:
        asyncio.create_task(_persist_clusters(clusters))

    return {"clusters": clusters}


async def _persist_clusters(clusters: dict):
    """Persiste clusters a Supabase en background."""
    try:
        await sb.upload_clusters(clusters)
    except Exception as e:
        logger.warning("Error persistiendo clusters a Supabase: %s", e)
