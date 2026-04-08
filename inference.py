"""
inference.py - Baseline agent for Meta OpenEnv Hackathon Submission.
"""
import os
import json
import httpx
from openai import OpenAI
from data_triage_env.client import DataTriageClient
from data_triage_env.models import DataAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

oai = OpenAI(api_key=HF_TOKEN or "", base_url=API_BASE_URL) if HF_TOKEN else None

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "take_action",
            "description": "Take a data cleaning action",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "cast", "impute", "dedupe", "rescale"]},
                    "column": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["action"],
            }
        }
    }
]

def run_task(task_id: str, max_steps: int) -> float:
    print(f"START")
    with DataTriageClient() as env:
        try:
            session_id, obs = env.reset(task_id, seed=42)
        except Exception as e:
            print(f"RESET_ERROR: {e}")
            print("END")
            return 0.0

        if oai is None:
            # Return safely when token is unavailable; never crash the script.
            print("MODEL_UNAVAILABLE: HF_TOKEN is missing")
            print("END")
            return 0.0

        messages = [
            {"role": "system", "content": (
                "You are a data cleaning agent. You have access to a dirty DataFrame. "
                "Your goal is to fix all data quality issues: nulls, type mismatches, "
                "duplicates, unit mismatches, and date format issues. "
                "Always inspect first, then act. Do NOT modify columns that look fine. "
                "Use the take_action function."
            )},
            {"role": "user", "content": f"Dataset info:\n{json.dumps(obs.model_dump(), indent=2)}\n\nClean this dataset."}
        ]
        last_score = 0.0
        
        for step in range(max_steps):
            try:
                response = oai.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
            except Exception as e:
                print(f"MODEL_ERROR: {e}")
                break
            
            if getattr(msg, "tool_calls", None) is None:
                # If no tool calls, the agent thinks it's done.
                break
                
            messages.append(msg)
            
            for tc in msg.tool_calls:
                try:
                    args_dict = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": "Invalid tool arguments JSON"}),
                    })
                    continue

                action_type = args_dict.get("action", "inspect")
                print("STEP")

                try:
                    action = DataAction(
                        action=action_type,
                        column=args_dict.get("column"),
                        params=args_dict.get("params", {}),
                    )
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Invalid action payload: {e}"}),
                    })
                    continue

                try:
                    obs, reward = env.step(session_id, action)
                except httpx.HTTPStatusError as e:
                    # Keep the episode alive and feed server-side validation errors
                    # back to the model so it can self-correct next step.
                    detail = e.response.text if e.response is not None else str(e)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Action rejected: {detail}"}),
                    })
                    continue

                last_score = reward.score
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"score": reward.score, "obs": obs.model_dump()}),
                })
                
                if reward.done:
                    print(f"END")
                    return last_score
                    
        print(f"END")
        return last_score

if __name__ == "__main__":
    try:
        for task_name in ["easy", "medium", "hard"]:
            max_s = {"easy": 20, "medium": 40, "hard": 60}[task_name]
            score = run_task(task_name, max_s)
            print(f"{task_name} score: {score:.4f}")
    except Exception as e:
        # Final guardrail: never crash with non-zero exit due to unexpected errors.
        print(f"FATAL_ERROR: {e}")
