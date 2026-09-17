import json

import httpx
import pytest

from scout.tools import pennsieve_discover, pennsieve_inspect
from scout import pipeline
from pennsieve_ai_utils.blackboard import Blackboard


def test_legacy_inspection_uses_public_assets_and_manifest(monkeypatch):
    manifest = {"files": [{"path": "files/assay/sample.fcs", "fileType": "FCS", "size": 10}]}
    monkeypatch.setattr(pennsieve_discover, "download_file", lambda detail, path: json.dumps(manifest).encode())
    monkeypatch.setattr(httpx, "get", lambda url, **kwargs: httpx.Response(200, text="Real README", request=httpx.Request("GET", url)))
    def no_s3(*args, **kwargs):
        pytest.fail("Public metadata must not require S3 credentials")
    monkeypatch.setattr(pennsieve_inspect, "get_object", no_s3)
    monkeypatch.setattr(pennsieve_inspect, "list_prefix", no_s3)
    result = pennsieve_inspect.inspect("s3://legacy-bucket/405/", {"id": 405, "version": 1, "readme": "https://example.com/readme.md"})
    assert result["readme"] == "Real README"
    assert result["manifest_file_count"] == 1
    assert "sample.fcs" in result["file_tree"]
    assert result["records"] == {}


def test_empty_seed_stops_before_llm_work(monkeypatch):
    monkeypatch.delenv("VP_DEGRADED_OK", raising=False)
    monkeypatch.setattr(pipeline, "_fetch_seed", lambda seed: ({"name": "seed"}, {}))
    bb = Blackboard("q", "pennsieve:405")
    with pytest.raises(RuntimeError, match="no inspectable files"):
        pipeline.run_scout(bb, object(), "pennsieve:405")
    assert bb.dataset_summaries[0].id == "pennsieve:405"
    assert bb.notebook[-1]["action"] == "blocked"


def test_download_failure_does_not_log_signed_url(monkeypatch):
    def denied(*args):
        raise httpx.HTTPStatusError("secret signed URL", request=httpx.Request("GET", "https://example.com?secret=xyz"), response=httpx.Response(403))
    monkeypatch.setattr(pennsieve_discover, "download_file", denied)
    monkeypatch.setattr(pennsieve_inspect, "fetch_readme", lambda uri: "")
    monkeypatch.setattr(pennsieve_inspect, "fetch_manifest", lambda uri: None)
    monkeypatch.setattr(pennsieve_inspect, "fetch_metadata_records", lambda *args, **kwargs: {})
    result = pennsieve_inspect.inspect("s3://legacy/405", {"id": 405, "version": 1})
    assert result["diagnostics"][0]["status"] == 403
    assert "secret" not in json.dumps(result)
