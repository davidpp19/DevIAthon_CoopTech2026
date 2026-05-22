"""
CoopTech Backend — Router Health Check.
"""

from fastapi import APIRouter
from config import API_CONFIG
from orchestrator.engine import pipeline_engine
from schemas.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check del servicio."""
    return HealthResponse(
        status="ok",
        version=API_CONFIG["version"],
        agents_count=len(pipeline_engine.agents),
        pipeline_status=pipeline_engine.status,
        models_ready=sum(1 for a in pipeline_engine.agents.values() if a.status == "ready"),
    )
