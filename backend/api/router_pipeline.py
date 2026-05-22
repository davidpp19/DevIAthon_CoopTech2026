"""
CoopTech Backend — Router del Pipeline.

Endpoints para controlar la ejecución del pipeline de datos y entrenamiento.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from orchestrator.engine import pipeline_engine
from schemas.schemas import PipelineStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run", response_model=PipelineStatusResponse)
async def run_pipeline(background_tasks: BackgroundTasks):
    """
    Ejecuta el pipeline completo: carga → limpieza → features → balanceo → entrenamiento.
    Se ejecuta en background y retorna inmediatamente con estado 'running'.
    """
    if pipeline_engine.status == "running":
        raise HTTPException(
            status_code=409,
            detail="El pipeline ya está en ejecución. Espere a que termine."
        )

    # Lanzar en background
    background_tasks.add_task(_run_pipeline_task)

    pipeline_engine.status = "running"
    pipeline_engine.current_phase = "queued"

    return PipelineStatusResponse(
        status="running",
        current_phase="queued",
        pipeline_duration_seconds=0,
        phase_durations={},
        agents={},
    )


async def _run_pipeline_task():
    """Tarea async que ejecuta el pipeline completo."""
    try:
        await pipeline_engine.run_full_pipeline()
    except Exception as e:
        logger.error(f"Pipeline falló: {e}", exc_info=True)


@router.get("/status", response_model=PipelineStatusResponse)
async def get_pipeline_status():
    """Retorna el estado actual del pipeline y todos los agentes."""
    status = pipeline_engine.get_status()
    return PipelineStatusResponse(**status)
