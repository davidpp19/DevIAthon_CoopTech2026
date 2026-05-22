"""
CoopTech Backend — Schemas Pydantic.

Modelos de request/response para la API REST.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ─── Schemas de Request ──────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Datos de un socio para predicción individual."""
    v_ah_cliente: Optional[float] = None
    saldo_disponible: float = Field(0, description="Saldo disponible en la cuenta")
    monto_bloq: float = Field(0, description="Monto bloqueado")
    ingresos: float = Field(0, description="Ingresos declarados")
    egresos: float = Field(0, description="Egresos declarados")
    edad: Optional[float] = None
    sexo: Optional[str] = None
    estado_civil: Optional[str] = None
    credito: int = Field(0, description="1 si tiene crédito activo")
    dias_sin_movimiento: float = Field(0)
    prod_bancario: Optional[float] = None
    oficina_cta: Optional[float] = None
    tipo_cuenta: Optional[str] = None
    cooplinea: int = 0
    tarjetas: int = 0
    v_fecha_nac: Optional[str] = None
    desviacion_maxima: float = 0
    tendencia_saldo: float = 0
    ratio_ingreso_egreso: float = 0
    flujo_neto: float = 0
    saldo_efectivo: float = 0
    ratio_bloqueo: float = 0
    antiguedad_cuenta_dias: float = 0
    productos_digitales: int = 0
    volumen_total: float = 0
    ratio_creditos_debitos: float = 0
    grupo_etario: float = 0


# ─── Schemas de Response ─────────────────────────────────────────────────────

class AgentStatus(BaseModel):
    """Estado de un agente."""
    agent_id: str
    agent_name: str
    status: str
    trained_at: Optional[str] = None
    training_duration_seconds: float = 0


class AgentResult(BaseModel):
    """Resultado de un agente."""
    agent_id: str
    agent_name: str
    status: str
    metrics: dict = {}
    results: dict = {}
    trained_at: Optional[str] = None


class PredictResponse(BaseModel):
    """Respuesta de predicción individual."""
    agent_id: str
    prediction: dict = {}


class PipelineStatusResponse(BaseModel):
    """Estado del pipeline."""
    status: str
    current_phase: str = ""
    pipeline_duration_seconds: float = 0
    phase_durations: dict = {}
    agents: dict = {}


class DashboardSummary(BaseModel):
    """Resumen consolidado del Dashboard."""
    pipeline_status: str
    pipeline_duration_seconds: float = 0
    phase_durations: dict = {}
    total_agents: int = 0
    agents_ready: int = 0
    agents_error: int = 0
    agents: dict = {}


class SocioProfile(BaseModel):
    """Perfil 360° de un socio."""
    v_ah_cliente: Any
    socio_data: dict = {}
    agents: dict = {}


class KPIResponse(BaseModel):
    """KPIs globales."""
    total_socios: int = 0
    tasa_riesgo_alto: float = 0
    socios_en_alerta: int = 0
    socios_sobreendeudados: int = 0
    menores_bloqueados: int = 0
    clusters_count: int = 0
    avg_credit_score: float = 0
    pipeline_status: str = "idle"
    # Campos adicionales para el frontend
    tasa_mora_pct: float = 0
    tasa_imputacion_ingresos_pct: float = 0
    menores_detectados: int = 0
    alertas_desvio: int = 0
    elegibles_pct: Optional[float] = None
    muestra_evaluada: Optional[int] = None
    distribucion_alerta: Optional[dict] = None
    distribucion_riesgo: Optional[dict] = None
    distribucion_canal: Optional[dict] = None


class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str = "ok"
    version: str
    agents_count: int
    pipeline_status: str
    models_ready: int = 0


class SegmentoProducto(BaseModel):
    """Segmento por producto."""
    prod_bancario: str
    n_socios: int
    tasa_mora_pct: float
    score_riesgo: Optional[float] = None


class DashboardSegmentos(BaseModel):
    """Segmentos del dashboard."""
    por_producto: list[SegmentoProducto]
    por_canal_cobranza: dict[str, int]


class ClienteInput(BaseModel):
    """Datos de entrada del socio para evaluación individual."""
    cliente_id: Optional[int] = None
    v_ah_cliente: Optional[int] = None
    edad: Optional[float] = None
    ingresos: Optional[float] = None
    egresos: Optional[float] = None
    saldo_disponible: Optional[float] = None
    estado_cta: Optional[str] = None
    prod_bancario: Optional[float] = None
    tipo_cuenta: Optional[str] = None
    estado_civil: Optional[str] = None
    v_fecha_nac: Optional[str] = None
    fecha_proceso: Optional[str] = None
    fecha_ultmov: Optional[str] = None
    dias_sin_movimiento: Optional[float] = None
    desviacion_maxima: Optional[float] = None
    tiene_alerta_desvio: Optional[int] = None
    dias_por_mes: Optional[str] = None
    cargas_familiares: Optional[int] = None
    flujo_neto: Optional[float] = None
    mora_30d: Optional[int] = None
    tiene_bloqueos: Optional[float] = None

    model_config = {"extra": "allow"}

    def to_agent_dict(self) -> dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if self.cliente_id is not None:
            d.setdefault("v_ah_cliente", self.cliente_id)
        return d


class ScoreResponse(BaseModel):
    """Resultado del score de 9 agentes consolidado."""
    cliente_id: Optional[Any] = None
    riesgo_global: float
    elegibilidad_credito: bool
    canal_cobranza: Optional[str] = None
    dia_pago_sugerido: Optional[int] = None
    alertas_activas: list[str] = []
    acciones_priorizadas: list[str] = []
    bloqueos: list[str] = []
    agentes: dict = {}

