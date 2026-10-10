"""Helpers shared by test files."""

import httpx2


def assert_problem(response: httpx2.Response, status: int) -> None:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body.keys() == {"type", "title", "status", "detail"}
    assert body["status"] == status
    assert "Traceback" not in response.text
