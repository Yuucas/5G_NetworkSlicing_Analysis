"""
Unit tests for API routes.
"""

import pytest

# Only import if FastAPI available
try:
    from fastapi.testclient import TestClient
    from src.api.app import app
    from src.api.routes import health, prediction, allocation, monitoring
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="Requires fastapi")
class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_readiness_check(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data
        assert "checks" in data

    def test_liveness_check(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["alive"] is True


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="Requires fastapi")
class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_predict_resource_demand(self, client):
        """Test resource demand prediction endpoint."""
        request_data = {
            "network_states": [
                {
                    "signal_strength_dbm": -75.0,
                    "latency_ms": 30.0,
                    "required_bandwidth_mbps": 10.0,
                    "application_type": "Video_Call",
                    "user_id": "User_1"
                }
            ],
            "model_type": "dqn",
            "horizon": 1
        }

        response = client.post("/api/v1/predict/resource_demand", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "model_used" in data
        assert len(data["predictions"]) == 1

    def test_forecast_bandwidth(self, client):
        """Test bandwidth forecasting endpoint."""
        response = client.post(
            "/api/v1/predict/bandwidth_forecast",
            params={"user_id": "User_1", "horizon_hours": 24}
        )
        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "forecast" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="Requires fastapi")
class TestAllocationEndpoints:
    """Test allocation endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_allocate_resources(self, client):
        """Test resource allocation endpoint."""
        request_data = {
            "user_id": "User_1",
            "application_type": "Video_Call",
            "signal_strength_dbm": -75.0,
            "latency_ms": 30.0,
            "required_bandwidth_mbps": 10.0,
            "priority": 7
        }

        response = client.post("/api/v1/allocate/request", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "user_id" in data
        assert "allocated_bandwidth_mbps" in data
        assert "allocation_percentage" in data
        assert data["user_id"] == "User_1"

    def test_allocate_emergency_service(self, client):
        """Test allocation for emergency service."""
        request_data = {
            "user_id": "Emergency_1",
            "application_type": "Emergency_Service",
            "signal_strength_dbm": -70.0,
            "latency_ms": 10.0,
            "required_bandwidth_mbps": 2.0,
            "priority": 10
        }

        response = client.post("/api/v1/allocate/request", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["priority_level"] == 10
        assert data["estimated_qos"] == "EXCELLENT"

    def test_batch_allocate(self, client):
        """Test batch allocation."""
        requests = [
            {
                "user_id": f"User_{i}",
                "application_type": "Video_Call",
                "signal_strength_dbm": -75.0,
                "latency_ms": 30.0,
                "required_bandwidth_mbps": 10.0,
            }
            for i in range(5)
        ]

        response = client.post("/api/v1/allocate/batch_allocate", json=requests)
        assert response.status_code == 200

        data = response.json()
        assert "allocations" in data
        assert len(data["allocations"]) == 5

    def test_optimize_resources(self, client):
        """Test resource optimization endpoint."""
        response = client.get("/api/v1/allocate/optimize")
        assert response.status_code == 200

        data = response.json()
        assert "algorithm" in data
        assert "objectives" in data


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="Requires fastapi")
class TestMonitoringEndpoints:
    """Test monitoring endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_get_metrics(self, client):
        """Test current metrics endpoint."""
        response = client.get("/api/v1/monitor/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "network_utilization" in data
        assert "active_connections" in data

    def test_get_performance_history(self, client):
        """Test performance history endpoint."""
        response = client.get("/api/v1/monitor/performance/history?hours=24")
        assert response.status_code == 200

        data = response.json()
        assert "period_hours" in data
        assert "metrics" in data

    def test_get_alerts(self, client):
        """Test alerts endpoint."""
        response = client.get("/api/v1/monitor/alerts")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)

    def test_get_sla_compliance(self, client):
        """Test SLA compliance endpoint."""
        response = client.get("/api/v1/monitor/sla_compliance")
        assert response.status_code == 200

        data = response.json()
        assert "overall_compliance_rate" in data
        assert "violations" in data

    def test_get_model_performance(self, client):
        """Test model performance endpoint."""
        response = client.get("/api/v1/monitor/model_performance")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "timestamp" in data
