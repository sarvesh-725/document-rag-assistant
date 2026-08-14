"""Real-service end-to-end coverage for phases 57-60.

Run with a live API, PostgreSQL, Redis, Qdrant, publisher, and Taskiq worker:
    INTEGRATION_API_URL=http://localhost:8000 pytest -q tests/test_phase57_60_integration.py
"""

import asyncio
import os
import uuid

import httpx
import pytest


API_URL = os.getenv("INTEGRATION_API_URL")
pytestmark = pytest.mark.skipif(
    not API_URL,
    reason="Set INTEGRATION_API_URL with all external services running",
)


async def _login(client: httpx.AsyncClient) -> str:
    username = f"integration_{uuid.uuid4().hex}"
    password = "integration-password"
    signup = await client.post("/api/v1/auth/signup", json={"username": username, "password": password})
    assert signup.status_code == 201, signup.text
    login = await client.post("/api/v1/auth/login", data={"username": username, "password": password})
    assert login.status_code == 200, login.text
    return login.json()["access_token"]


async def _query(client: httpx.AsyncClient, token: str, session_id: str, request_id: str, document_ids: list[str], question: str):
    return await client.post(
        "/api/v1/chat/query",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "session_id": session_id,
            "client_request_id": request_id,
            "question": question,
            "selected_document_ids": document_ids,
        },
        timeout=120,
    )


async def _wait_ready(client: httpx.AsyncClient, token: str, document_id: str):
    for _ in range(60):
        documents = (await client.get("/api/v1/documents", headers={"Authorization": f"Bearer {token}"})).json()
        item = next(item for item in documents if item["document_id"] == document_id)
        if item["status"] in {"READY", "FAILED"}:
            assert item["status"] == "READY", item
            return item
        await asyncio.sleep(1)
    pytest.fail("document did not reach READY")


@pytest.mark.asyncio
async def test_phases_57_to_60_real_document_duplicate_session_query_and_delete_flow():
    async with httpx.AsyncClient(base_url=API_URL) as client:
        token = await _login(client)
        auth = {"Authorization": f"Bearer {token}"}
        session_a = (await client.post("/api/v1/sessions", headers=auth, json={"title": "A"})).json()["session_id"]
        session_b = (await client.post("/api/v1/sessions", headers=auth, json={"title": "B"})).json()["session_id"]

        uploads = []
        for content in (b"%PDF-1.4\nrevenue was 10", b"%PDF-1.4\nrevenue was 20", b"%PDF-1.4\nrevenue was 30"):
            response = await client.post(
                "/api/v1/documents",
                headers=auth,
                files={"file": ("report.pdf", content, "application/pdf")},
            )
            assert response.status_code == 202, response.text
            uploads.append(response.json())
        assert len({item["document_id"] for item in uploads}) == 3

        await _wait_ready(client, token, uploads[0]["document_id"])
        documents = (await client.get("/api/v1/documents", headers=auth)).json()
        names = [item["display_name"] for item in documents if item["document_id"] in {upload["document_id"] for upload in uploads}]
        assert names == ["report.pdf", "report (2).pdf", "report (3).pdf"]

        selected = [uploads[0]["document_id"]]
        first, second = await asyncio.gather(
            _query(client, token, session_a, "same-request", selected, "What was revenue?"),
            _query(client, token, session_a, "same-request", selected, "What was revenue?"),
        )
        assert first.status_code == second.status_code == 200
        assert first.headers.get("content-type", "").startswith("text/event-stream")
        assert (await _query(client, token, session_b, str(uuid.uuid4()), selected, "Summarize revenue.")).status_code == 200

        race_task = asyncio.create_task(_query(client, token, session_a, str(uuid.uuid4()), selected, "What was revenue?"))
        delete = await client.delete(f"/api/v1/documents/{uploads[0]['document_id']}", headers=auth)
        assert delete.status_code == 200
        race = await race_task
        assert race.status_code in {200, 400}
        documents = (await client.get("/api/v1/documents", headers=auth)).json()
        assert uploads[0]["document_id"] not in {item["document_id"] for item in documents}
