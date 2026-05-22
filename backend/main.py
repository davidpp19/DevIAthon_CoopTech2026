"""
CoopTech Backend — Entry Point.

Predictor de Comportamiento de Pago de Socios.
Cooperative de Ahorro y Crédito Tulcán — DevIAthon 2026.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Agregar el directorio backend al path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_CONFIG
from api.router_pipeline import router as pipeline_router
from api.router_agents import router as agents_router
from api.router_dashboard import router as dashboard_router
from api.router_health import router as health_router
from api.router_score import router as score_router
from orchestrator.engine import pipeline_engine

# ─── Configuración de Logging ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("cooptech")

# ─── Crear la aplicación FastAPI ─────────────────────────────────────────────
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS (permitir acceso desde cualquier frontend en desarrollo) ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, restringir a dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Montar Routers ─────────────────────────────────────────────────────────
PREFIX = API_CONFIG["prefix"]

app.include_router(health_router)
app.include_router(health_router, prefix=PREFIX)
app.include_router(pipeline_router, prefix=PREFIX)
app.include_router(agents_router, prefix=PREFIX)
app.include_router(dashboard_router, prefix=PREFIX)
app.include_router(score_router, prefix=PREFIX)


# ─── Evento de inicio ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info(f"  {API_CONFIG['title']}")
    logger.info(f"  Version: {API_CONFIG['version']}")
    logger.info("=" * 70)
    logger.info("Servidor iniciado. Endpoints disponibles:")
    logger.info("  Docs:      http://localhost:8000/docs")
    logger.info("  Health:    http://localhost:8000/health")
    logger.info(f"  Pipeline:  http://localhost:8000{PREFIX}/pipeline/run")
    logger.info(f"  Agentes:   http://localhost:8000{PREFIX}/agents")
    logger.info(f"  Dashboard: http://localhost:8000{PREFIX}/dashboard/summary")
    logger.info(f"  Scoring:   http://localhost:8000{PREFIX}/score/cliente/{{id}}")
    logger.info("=" * 70)

    if pipeline_engine.status == "idle":
        logger.info("Iniciando entrenamiento del pipeline automaticamente...")
        asyncio.create_task(pipeline_engine.run_full_pipeline())


# ─── Para ejecutar directamente ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
