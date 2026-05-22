"""
CoopTech Backend — Router de Agentes.

Endpoints para consultar resultados y ejecutar predicciones por agente.
"""

import logging

from fastapi import APIRouter, HTTPException

from orchestrator.engine import pipeline_engine
from schemas.schemas import AgentStatus, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agentes"])


@router.get("", response_model=list[AgentStatus])
async def list_agents():
    """Lista todos los agentes con su estado actual."""
    return [
        AgentStatus(**agent.get_status())
        for agent in pipeline_engine.agents.values()
    ]


@router.get("/{agent_id}/results")
async def get_agent_results(agent_id: str):
    """Retorna los resultados del último entrenamiento de un agente."""
    agent = pipeline_engine.agents.get(agent_id)
    if agent is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agente '{agent_id}' no encontrado. "
                   f"Agentes disponibles: {list(pipeline_engine.agents.keys())}",
        )

    if agent.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Agente '{agent_id}' no está listo (estado: {agent.status}). "
                   "Ejecute el pipeline primero con POST /api/v1/pipeline/run",
        )

    return {
        "agent_id": agent.agent_id,
        "agent_name": agent.agent_name,
        "status": agent.status,
        "metrics": agent.metrics,
        "results": agent.results,
        "trained_at": agent.trained_at.isoformat() if agent.trained_at else None,
    }


@router.get("/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str):
    """Retorna las métricas de rendimiento del modelo."""
    agent = pipeline_engine.agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' no encontrado.")

    if agent.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Agente '{agent_id}' no está listo (estado: {agent.status}).",
        )

    return {
        "agent_id": agent.agent_id,
        "metrics": agent.metrics,
        "training_duration_seconds": agent.training_duration_seconds,
    }


@router.post("/{agent_id}/predict", response_model=PredictResponse)
async def predict(agent_id: str, request: PredictRequest):
    """Ejecuta una predicción individual para un socio en un agente específico."""
    agent = pipeline_engine.agents.get(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' no encontrado.")

    if agent.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Agente '{agent_id}' no está listo (estado: {agent.status}).",
        )

    try:
        input_data = request.model_dump()
        result = await agent.predict(input_data)
        return PredictResponse(agent_id=agent_id, prediction=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")
