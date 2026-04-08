import pytest
from fastapi.testclient import TestClient
from data_triage_env.server import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_reset_easy():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert "observation" in data
    assert data["observation"]["shape"][0] == 200

def test_full_episode_easy():
    r = client.post("/reset", json={"task_id": "easy", "seed": 99})
    sid = r.json()["session_id"]
    # Inspect
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    assert r2.status_code == 200
    reward = r2.json()["reward"]
    assert 0.0 <= reward["score"] <= 1.0

def test_invalid_action_missing_column():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "cast"}})
    assert r2.status_code == 422

def test_session_not_found():
    r = client.post("/step", json={"session_id": "fake-id", "action": {"action": "inspect"}})
    assert r.status_code == 404

def test_score_is_deterministic():
    r1 = client.post("/reset", json={"task_id": "easy", "seed": 42})
    sid1 = r1.json()["session_id"]
    r2 = client.post("/reset", json={"task_id": "easy", "seed": 42})
    sid2 = r2.json()["session_id"]
    # Both sessions with same seed should produce same initial observation
    obs1 = r1.json()["observation"]
    obs2 = r2.json()["observation"]
    assert obs1["shape"] == obs2["shape"]

def test_reward_bounds():
    """Score must always be in [0.0, 1.0]"""
    r = client.post("/reset", json={"task_id": "hard", "seed": 7})
    sid = r.json()["session_id"]
    for _ in range(5):
        r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
        if r2.status_code == 200:
            score = r2.json()["reward"]["score"]
            assert 0.0 <= score <= 1.0
