import json
from data_triage_env.client import DataTriageClient
from data_triage_env.models import DataAction

def run_simple_agent(task_id: str):
    print(f"\n--- Running Simple Agent on {task_id.upper()} ---")
    with DataTriageClient() as env:
        session_id, obs = env.reset(task_id, seed=42)
        
        # 1. Dedupe immediately
        obs, reward = env.step(session_id, DataAction(action="dedupe"))
        
        # 2. Iterate through columns and apply heuristic fixes
        for col_stat in obs.columns:
            col = col_stat.name
            
            # Rule 1: Fix numeric strings (e.g., in revenue or sensor_id)
            if col_stat.dtype == "object":
                # Look for currency symbols, prefixes, or "N/A" markers
                sample_str = str(col_stat.sample_values[0])
                if "USD" in sample_str or "," in sample_str:
                    env.step(session_id, DataAction(action="cast", column=col, params={"dtype": "float64"}))
                elif "SENSOR_" in sample_str:
                    env.step(session_id, DataAction(action="cast", column=col, params={"dtype": "int64"}))
                elif "N/A" in [str(v) for v in col_stat.sample_values]:
                    env.step(session_id, DataAction(action="cast", column=col, params={"dtype": "float64"}))
                elif "/" in sample_str or "-" in sample_str:
                     # Looks like a date - cast to datetime (ISO 8601)
                     env.step(session_id, DataAction(action="cast", column=col, params={"dtype": "datetime64[ns]"}))

            # Rule 2: Impute missing values
            if col_stat.null_count > 0:
                env.step(session_id, DataAction(action="impute", column=col, params={"strategy": "median"}))

        # Final check
        obs, reward = env.step(session_id, DataAction(action="inspect"))
        print(f"Final Score for {task_id}: {reward.score:.4f}")
        return reward.score

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_simple_agent(task)
