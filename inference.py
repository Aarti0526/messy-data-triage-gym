import os
import sys
import json
import httpx
from openai import OpenAI
from data_triage_env.client import DataTriageClient
from data_triage_env.models import DataAction

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
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

def run_task(task_id: str, max_steps: int):
    # One [START] line at episode begin.
    print(f"[START] task={task_id} env=messy-data-triage-gym model={MODEL_NAME}", flush=True)
    
    steps_taken = 0
    rewards = []
    success = False
    
    with DataTriageClient() as env:
        try:
            session_id, obs = env.reset(task_id, seed=42)
        except Exception as e:
            # Emitting an end immediately on failure during reset
            print(f"[END] success=false steps=0 rewards=", flush=True)
            return

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

        for step in range(max_steps):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=TOOLS,
                )
                msg = response.choices[0].message
            except Exception as e:
                # Emitting an error step if model completion fails
                break

            if getattr(msg, "tool_calls", None) is None:
                break

            messages.append(msg)

            for tc in msg.tool_calls:
                steps_taken += 1
                action_type = "unknown"
                column = ""
                action_str = "unknown"

                try:
                    args_dict = json.loads(tc.function.arguments or "{}")
                    action_type = args_dict.get("action", "inspect")
                    column = args_dict.get("column", "")
                    
                    if column:
                        action_str = f"{action_type}('{column}')"
                    else:
                        action_str = f"{action_type}()"
                    # Remove spaces to comply with requirements if any
                    action_str = action_str.replace(" ", "")

                    action = DataAction(
                        action=action_type,
                        column=args_dict.get("column"),
                        params=args_dict.get("params", {}),
                    )
                    
                    obs, reward = env.step(session_id, action)
                    
                    r_val = float(reward.score)
                    r_str = f"{r_val:.2f}"
                    rewards.append(r_str)
                    
                    d_str = "true" if reward.done else "false"
                    print(f"[STEP] step={steps_taken} action={action_str} reward={r_str} done={d_str} error=null", flush=True)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"score": reward.score, "obs": obs.model_dump()}),
                    })
                    
                    if reward.done:
                        success = True
                        r_joined = ",".join(rewards)
                        s_str = "true" if success else "false"
                        print(f"[END] success={s_str} steps={steps_taken} rewards={r_joined}", flush=True)
                        return

                except httpx.HTTPStatusError as e:
                    detail = e.response.text if e.response is not None else str(e)
                    detail = detail.replace('\n', ' ').replace('\r', '')
                    r_str = "0.00"
                    rewards.append(r_str)
                    print(f"[STEP] step={steps_taken} action={action_str} reward={r_str} done=false error={detail}", flush=True)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Action rejected: {detail}"}),
                    })
                except Exception as e:
                    err_msg = str(e).replace('\n', ' ').replace('\r', '')
                    if err_msg == "":
                        err_msg = type(e).__name__
                    r_str = "0.00"
                    rewards.append(r_str)
                    print(f"[STEP] step={steps_taken} action={action_str} reward={r_str} done=false error={err_msg}", flush=True)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Error: {err_msg}"}),
                    })

        # Outside the loop max steps reached
        r_joined = ",".join(rewards)
        s_str = "true" if success else "false"
        print(f"[END] success={s_str} steps={steps_taken} rewards={r_joined}", flush=True)

if __name__ == "__main__":
    for task_name in ["easy", "medium", "hard"]:
        max_s = {"easy": 20, "medium": 40, "hard": 60}[task_name]
        run_task(task_name, max_s)
