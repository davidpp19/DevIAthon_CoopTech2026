"""
CoopTech Backend — Tests de la API REST.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Agregar backend al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["agents_count"] == 9
        assert "version" in data

    def test_docs_available(self):
        response = client.get("/docs")
        assert response.status_code == 200


class TestPipelineEndpoints:
    def test_pipeline_status_idle(self):
        response = client.get("/api/v1/pipeline/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["idle", "running", "ready"]

    def test_pipeline_status_has_agents(self):
        response = client.get("/api/v1/pipeline/status")
        data = response.json()
        assert "agents" in data


class TestAgentEndpoints:
    def test_list_agents(self):
        response = client.get("/api/v1/agents")
        assert response.status_code == 200
        agents = response.json()
        assert len(agents) == 9
        # Verificar que cada agente tiene los campos requeridos
        for agent in agents:
            assert "agent_id" in agent
            assert "agent_name" in agent
            assert "status" in agent

    def test_agent_not_found(self):
        response = client.get("/api/v1/agents/nonexistent/results")
        assert response.status_code == 404

    def test_agent_not_trained(self):
        response = client.get("/api/v1/agents/credit_scoring/results")
        assert response.status_code == 400  # No entrenado aún


class TestDashboardEndpoints:
    def test_dashboard_summary(self):
        response = client.get("/api/v1/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_status" in data
        assert "total_agents" in data
        assert data["total_agents"] == 9

    def test_dashboard_kpis(self):
        response = client.get("/api/v1/dashboard/kpis")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline_status" in data

    def test_socio_profile_not_ready(self):
        response = client.get("/api/v1/dashboard/socios/349.0")
        assert response.status_code == 400  # Pipeline no listo
