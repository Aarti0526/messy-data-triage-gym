from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

ActionType = Literal["inspect", "cast", "impute", "dedupe", "rescale"]

class DataAction(BaseModel):
    action: ActionType
    column: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "DataAction":
        if self.action in ("cast", "impute", "rescale") and not self.column:
            raise ValueError(f"action '{self.action}' requires 'column'")
        return self

class ColumnStats(BaseModel):
    name: str
    dtype: str
    null_count: int
    sample_values: list[Any]
    unique_count: int

class DataObservation(BaseModel):
    step: int
    columns: list[ColumnStats]
    shape: tuple[int, int]
    message: str = ""

class DataReward(BaseModel):
    reward: float
    score: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)

class ResetRequest(BaseModel):
    task_id: Literal["easy", "medium", "hard"]
    seed: Optional[int] = None

class StepRequest(BaseModel):
    session_id: str
    action: DataAction
