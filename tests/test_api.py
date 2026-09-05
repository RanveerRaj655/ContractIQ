"""
test_api.py
-----------
Tests for the FastAPI service using TestClient.
Mocks the RetrievalPipeline and LLMClient to avoid actual inference or API calls.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

# Make sure we can import from api
sys.path.insert(0, str(Path(__file__).parent.parent))

from contractiq.chunking import Chunk

# We mock the entire `api.main.RetrievalPipeline` and `api.main.LLMClient`
# BEFORE importing `app` so that the startup event uses the mocks.
patcher_pipeline = patch("api.main.RetrievalPipeline")
patcher_llm = patch("api.main.LLMClient")
patcher_halluc = patch("api.main.check_hallucination", return_value={"verdict": "SUPPORTED", "reason": "Mocked"})

MockPipelineCls = patcher_pipeline.start()
MockLLMCls = patcher_llm.start()
MockHalluc = patcher_halluc.start()

from api.main import app

client = TestClient(app)

def teardown_module(module):
    patcher_pipeline.stop()
    patcher_llm.stop()
    patcher_halluc.stop()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint():
    # Setup mock returns
    mock_pipeline_inst = MockPipelineCls.return_value
    mock_llm_inst = MockLLMCls.return_value

    # Mock retrieve to return a high score (no abstention)
    fake_chunk = Chunk(doc_id="test_doc.txt", start=0, end=10, text="Mock text")
    mock_pipeline_inst.retrieve.return_value = [(fake_chunk, 10.0)]
    
    # Mock LLM generation
    mock_llm_inst.generate_answer.return_value = "This is a mock answer."
    
    # Test the endpoint (TestClient automatically triggers the startup event)
    with TestClient(app) as live_client:
        response = live_client.post("/query", json={"question": "What is the governing law?"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["answer"] == "This is a mock answer."
        assert data["abstained"] is False
        assert data["hallucination_verdict"] == "SUPPORTED"
        assert "latency_ms" in data
        assert len(data["sources"]) == 1
        assert data["sources"][0]["doc_id"] == "test_doc.txt"


def test_query_rate_limit():
    with TestClient(app) as live_client:
        # Fire 20 fast requests to hit the limit
        for _ in range(20):
            response = live_client.post("/query", json={"question": "Rate limit test"})
            
        # The 21st should 429
        response_429 = live_client.post("/query", json={"question": "Should be blocked"})
        assert response_429.status_code == 429
        assert "Rate limit exceeded" in response_429.json()["detail"]
