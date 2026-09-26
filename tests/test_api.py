"""API duman testi: model yüklenir, örnek şirket verisiyle tahmin döner (CPU)."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web_app"))
from main import app  # noqa: E402

client = TestClient(app)


def test_features_and_prediction():
    feats = client.get("/api/features").json()
    assert feats["count"] == len(feats["features"]) > 0
    for kind in ("bankrupt", "non_bankrupt", "mean"):
        sample = client.get("/api/sample", params={"type": kind}).json()["features"]
        r = client.post("/predict", json={"features": sample})
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in (0, 1) and 0.0 <= body["probability"] <= 1.0


def test_invalid_payload_is_rejected():
    assert client.post("/predict", json={"features": {}}).status_code == 400
