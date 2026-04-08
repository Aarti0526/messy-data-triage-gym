"""
inference.py - Baseline agent for Meta OpenEnv Hackathon Submission.
"""
import os
import json
from openai import OpenAI
from data_triage_env.client import DataTriageClient
from data_triage_env.models import DataAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

if not HF_TOKEN:
    print("HF_TOKEN is missing!")
    exit(1)

oai = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL
)

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
        session_id, obs = env.reset(task_id, seed=42)
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
            response = oai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            
            if getattr(msg, "tool_calls", None) is None:
                # If no tool calls, the agent thinks it's done.
                break
                
            messages.append(msg)
            
            for tc in msg.tool_calls:
                args_dict = json.loads(tc.function.arguments)
                action_type = args_dict.get("action", "inspect")
                print(f"STEP")
                
                action = DataAction(
                    action=action_type,
                    column=args_dict.get("column"),
                    params=args_dict.get("params", {}),
                )
                obs, reward = env.step(session_id, action)
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
    for task_name in ["easy", "medium", "hard"]:
        max_s = {"easy": 20, "medium": 40, "hard": 60}[task_name]
        score = run_task(task_name, max_s)
