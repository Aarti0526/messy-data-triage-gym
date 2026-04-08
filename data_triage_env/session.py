import time
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from data_triage_env.engine.dataset_factory import generate_clean
from data_triage_env.engine.corruptor import GroundTruthManifest, CORRUPT_FNS
from data_triage_env.models import DataObservation
from data_triage_env.engine.executor import _observe

@dataclass
class EpisodeState:
    session_id: str
    task_id: str
    current_df: pd.DataFrame
    manifest: GroundTruthManifest
    step: int = 0
    max_steps: int = 20
    done: bool = False
    created_at: float = field(default_factory=time.time)

MAX_STEPS = {"easy": 20, "medium": 40, "hard": 60}

class SessionManager:
    def __init__(self):
        self._sessions: dict[str, EpisodeState] = {}

    def _cleanup(self):
        now = time.time()
        expired = [sid for sid, s in self._sessions.items() if now - s.created_at > 3600]
        for sid in expired:
            self._sessions.pop(sid, None)

    def create(self, task_id: str, seed: int | None = None) -> tuple[str, DataObservation]:
        self._cleanup()
        if seed is None:
            seed = np.random.randint(0, 2**31)
        rng = np.random.default_rng(seed)
        clean_df = generate_clean(task_id, seed)
        dirty_df, manifest = CORRUPT_FNS[task_id](clean_df, rng)
        session_id = str(uuid.uuid4())
        state = EpisodeState(
            session_id=session_id,
            task_id=task_id,
            current_df=dirty_df,
            manifest=manifest,
            max_steps=MAX_STEPS[task_id],
        )
        self._sessions[session_id] = state
        obs = _observe(dirty_df)
        obs.step = 0
        return session_id, obs

    def get(self, session_id: str) -> EpisodeState:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found")
        return self._sessions[session_id]

    def update_df(self, session_id: str, new_df: pd.DataFrame, obs: DataObservation):
        state = self._sessions[session_id]
        state.current_df = new_df
        state.step += 1
        if state.step >= state.max_steps:
            state.done = True
        obs.step = state.step

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)
