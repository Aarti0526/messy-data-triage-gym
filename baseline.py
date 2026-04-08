"""
Baseline agent using Hugging Face Serverless Inference via OpenAI client drop-in.
Runs all 3 tasks and prints scores.
Usage: HF_TOKEN=hf_... python baseline.py
"""
import json
import os
from openai import OpenAI
from data_triage_env.client import DataTriageClient
from data_triage_env.models import DataAction

oai = OpenAI(
    api_key=os.environ.get("HF_TOKEN", ""),
    base_url="https://api-inference.huggingface.co/v1/"
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
                    "action": {"type": "string", "enum": ["inspect","cast","impute","dedupe","rescale"]},
                    "column": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["action"],
            }
        }
    }
]

def run_task(task_id: str, max_steps: int) -> float:
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
        for _ in range(max_steps):
            response = oai.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                action = DataAction(
                    action=args["action"],
                    column=args.get("column"),
                    params=args.get("params", {}),
                )
                obs, reward = env.step(session_id, action)
                last_score = reward.score
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"score": reward.score, "obs": obs.model_dump()}),
                })
                if reward.done:
                    return last_score
        return last_score

if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        print("Set HF_TOKEN to run baseline with Hugging Face Serverless.")
        exit(1)
    for task in ["easy", "medium", "hard"]:
        max_s = {"easy": 20, "medium": 40, "hard": 60}[task]
        score = run_task(task, max_s)
        print(f"Task {task}: score = {score:.4f}")
