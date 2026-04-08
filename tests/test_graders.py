import pytest
import pandas as pd
import numpy as np
from data_triage_env.graders.easy_grader import score as score_easy
from data_triage_env.graders.medium_grader import score as score_medium
from data_triage_env.graders.hard_grader import score as score_hard
from data_triage_env.engine.corruptor import GroundTruthManifest, CorruptionRecord

def test_hard_grader_trap_penalty():
    # Setup a clean DF and a manifest with a trap
    clean_df = pd.DataFrame({"pressure": [1.0, 2.0, 3.0], "val": [10, 20, 30]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[
            CorruptionRecord(corruption_type="trap", column="pressure", row_indices=[], original_values=[], expected_fix="none"),
            CorruptionRecord(corruption_type="null", column="val", row_indices=[0], original_values=[10], expected_fix="impute")
        ]
    )
    
    # Case 1: Perfect fix without touching trap
    agent_df = clean_df.copy()
    s = score_hard(agent_df, manifest)
    assert s == 1.0
    
    # Case 2: Fix bug but TOUCH trap
    agent_df_broken = clean_df.copy()
    agent_df_broken.at[0, "pressure"] = 9.9
    s = score_hard(agent_df_broken, manifest)
    # Penalty multiplier for hard is 20.0. 
    # Broken clean = 1 cell. total_bugs = 1.
    # penalty = min(1.0, (1 * 20) / 1) = 1.0
    # raw = 1/1 = 1.0. 
    # score = 1.0 - 1.0 = 0.0
    # Also capped at 0.5 because trap_broken.
    assert s == 0.0

def test_medium_grader_penalty():
    clean_df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column="col", row_indices=[0], original_values=[1], expected_fix="impute")]
    )
    
    # Break 1 clean cell (row 1)
    agent_df = clean_df.copy()
    agent_df.at[1, "col"] = 99
    
    # Medium penalty is 15x
    s = score_medium(agent_df, manifest)
    # total_bugs = 1
    # raw = 1 (bug at row 0 is "fixed" because we started with clean data)
    # penalty = (1 * 15) / 1 = 15.0 (capped at 1.0)
    # score = 1.0 - 1.0 = 0.0
    assert s == 0.0

def test_easy_grader_penalty():
    clean_df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column="col", row_indices=[0], original_values=[1], expected_fix="impute")]
    )
    
    # Break 1 clean cell
    agent_df = clean_df.copy()
    agent_df.at[1, "col"] = 99
    
    # Easy penalty is 10x
    s = score_easy(agent_df, manifest)
    assert s == 0.0
