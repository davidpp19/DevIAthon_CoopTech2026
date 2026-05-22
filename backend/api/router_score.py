"""
CoopTech Backend — Router de Scoring.

Endpoints para evaluar individualmente a un socio.
Persiste resultados en Supabase.
"""

import asyncio
import logging
from fastapi import APIRouter, HTTPException

from orchestrator.engine import pipeline_engine
from schemas.schemas import ClienteInput, ScoreResponse
import supabase_client as sb

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/score", tags=["Scoring"])


@router.post("/cliente", response_model=ScoreResponse)
async def score_cliente(body: ClienteInput):
    """
    Evalúa a un socio con los datos provistos en el body.
    """
    if pipeline_engine.status != "ready":
        raise HTTPException(
            status_code=503,
            detail="El pipeline no está listo. Los modelos deben ser entrenados primero."
        )

    data = body.to_agent_dict()
    # Si no tiene cliente_id pero tiene v_ah_cliente o viceversa
    cid = data.get("v_ah_cliente") or data.get("cliente_id")
    if cid is not None:
        try:
            # Intentar rellenar con datos existentes si solo se pasó el ID
            existing_data = pipeline_engine.get_client_data(float(cid))
            # Mezclar existentes con nuevos
            existing_data.update(data)
            data = existing_data
        except Exception:
            # Si no existe en la base, continuar con lo provisto
            pass

    try:
        res = await pipeline_engine.run_client_scoring(data)
        # Persistir en Supabase en background
        asyncio.create_task(_persist_score(res))
        return ScoreResponse(**res)
    except Exception as e:
        logger.error(f"Error al evaluar cliente: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cliente/{cliente_id}", response_model=ScoreResponse)
async def score_cliente_by_id(cliente_id: float):
    """
    Evalúa a un socio existente en la base de datos (df_featured) por su ID.
    """
    if pipeline_engine.status != "ready":
        raise HTTPException(
            status_code=503,
            detail="El pipeline no está listo. Los modelos deben ser entrenados primero."
        )

    try:
        client_data = pipeline_engine.get_client_data(cliente_id)
        res = await pipeline_engine.run_client_scoring(client_data)
        # Persistir en Supabase en background
        asyncio.create_task(_persist_score(res))
        return ScoreResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error al evaluar cliente {cliente_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _persist_score(result: dict):
    """Persiste resultado de scoring a Supabase en background."""
    try:
        await sb.upload_scoring_result(result)
    except Exception as e:
        logger.warning("Error persistiendo scoring a Supabase: %s", e)
