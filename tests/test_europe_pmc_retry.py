"""Europe PMC intermittently 404s valid search URLs (throttle/WAF) — the same
query 404'd once and returned normally seconds later in the 2026-08-19 run."""
from __future__ import annotations

import httpx
import pytest

from scout.tools import europe_pmc


def _resp(status: int, json_body=None):
    req = httpx.Request("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search")
    return httpx.Response(status, request=req, json=json_body if json_body is not None else {})


def test_transient_404_is_retried(monkeypatch):
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return _resp(404)
        return _resp(200, {"resultList": {"result": []}})
    monkeypatch.setattr(europe_pmc.httpx, "get", fake_get)
    monkeypatch.setattr(europe_pmc.time, "sleep", lambda s: None)
    assert europe_pmc.search("seizure detection") == []
    assert len(calls) == 2


def test_persistent_404_finally_raises(monkeypatch):
    monkeypatch.setattr(europe_pmc.httpx, "get",
                        lambda url, params=None, timeout=None: _resp(404))
    monkeypatch.setattr(europe_pmc.time, "sleep", lambda s: None)
    with pytest.raises(httpx.HTTPStatusError):
        europe_pmc.search("anything")


def test_real_client_error_is_not_retried(monkeypatch):
    calls = []
    def fake_get(url, params=None, timeout=None):
        calls.append(1)
        return _resp(400)
    monkeypatch.setattr(europe_pmc.httpx, "get", fake_get)
    with pytest.raises(httpx.HTTPStatusError):
        europe_pmc.search("bad request")
    assert len(calls) == 1
