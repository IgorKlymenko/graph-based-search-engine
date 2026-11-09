import pytest
from fastapi.testclient import TestClient
from src.app import app

@pytest.fixture
def client(monkeypatch):
    """Fixture to provide a synchronous FastAPI TestClient."""
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "mysecret")
    with TestClient(app) as c:
        yield c