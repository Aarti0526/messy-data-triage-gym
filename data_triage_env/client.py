from __future__ import annotations
import httpx
from data_triage_env.models import DataAction, DataObservation, DataReward

class DataTriageClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str, seed: int | None = None) -> tuple[str, DataObservation]:
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = self._client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["session_id"], DataObservation(**data["observation"])

    def step(self, session_id: str, action: DataAction) -> tuple[DataObservation, DataReward]:
        payload = {"session_id": session_id, "action": action.model_dump()}
        r = self._client.post(f"{self.base_url}/step", json=payload)
        r.raise_for_status()
        data = r.json()
        return DataObservation(**data["observation"]), DataReward(**data["reward"])

    def health(self) -> dict:
        r = self._client.get(f"{self.base_url}/health")
        r.raise_for_status()
        return r.json()

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
