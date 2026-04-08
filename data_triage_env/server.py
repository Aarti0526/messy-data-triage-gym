from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from data_triage_env.models import ResetRequest, StepRequest, DataReward
from data_triage_env.session import SessionManager
from data_triage_env.engine.executor import run_action
from data_triage_env.graders.easy_grader import score as score_fn

logger = logging.getLogger("data_triage_env")
_manager = SessionManager()

GRADER_MAP = {
    "easy":   __import__("data_triage_env.graders.easy_grader",   fromlist=["score"]).score,
    "medium": __import__("data_triage_env.graders.medium_grader", fromlist=["score"]).score,
    "hard":   __import__("data_triage_env.graders.hard_grader",   fromlist=["score"]).score,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Messy Data Triage Gym started")
    yield
    logger.info("Shutting down")

app = FastAPI(title="Messy Data Triage Gym", version="0.1.0", lifespan=lifespan)

@app.get("/")
async def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
async def health():
    return {"status": "ok", "env": "messy-data-triage-gym-v1"}

@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    session_id, obs = _manager.create(req.effective_task, req.seed)
    return {"session_id": session_id, "observation": obs.model_dump(), "info": {"session_id": session_id}}

@app.post("/step")
async def step(req: StepRequest):
    try:
        state = _manager.get(req.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")

    if state.done:
        raise HTTPException(status_code=400, detail="Episode already finished")

    try:
        new_df, obs, msg = run_action(state.current_df, req.action)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    _manager.update_df(req.session_id, new_df, obs)
    obs.message = msg

    grader = GRADER_MAP[state.task_id]
    current_score = grader(new_df, state.manifest)

    reward = DataReward(
        reward=float(current_score),
        score=float(current_score),
        done=state.done,
        info={"step": state.step, "task": state.task_id, "message": msg}
    )
    return {"observation": obs.model_dump(), "reward": reward.model_dump()}

@app.get("/state/{session_id}")
async def get_state(session_id: str):
    try:
        state = _manager.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": state.session_id,
        "task_id": state.task_id,
        "step": state.step,
        "done": state.done,
        "shape": list(state.current_df.shape),
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
