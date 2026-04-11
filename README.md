---
title: Messy Data Triage Gym
emoji: 🧹
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
tags: [openenv, reinforcement-learning, data-cleaning, llm-agent, tabular-data]
---

<div align="center">

# 🧹 Messy Data Triage Gym

### *Can your LLM think like a Data Engineer?*

**An OpenEnv-compatible Reinforcement Learning environment that challenges AI agents to autonomously diagnose and repair real-world tabular data corruption — across three escalating difficulty tiers.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blueviolet)](https://github.com/meta-pytorch/OpenEnv)
[![Tests](https://img.shields.io/badge/Tests-81%2F81%20passing-brightgreen)](#testing)
[![Difficulty](https://img.shields.io/badge/Difficulty-Easy%20%7C%20Medium%20%7C%20Hard-orange)](#difficulty-tiers)
[![License](https://img.shields.io/badge/License-MIT-blue)](#)

</div>

---

## 🎯 What Is This?

**Messy Data Triage Gym** is a rigorous, stateful reinforcement learning environment that simulates the real chaos of data engineering pipelines. It stochastically injects 6+ classes of corruption into structured tabular datasets and challenges an LLM agent — equipped with 5 tool-bound actions — to systematically discover and repair every issue.

Unlike toy benchmarks, this gym tests **multi-step causal reasoning**: a wrong action at step 3 can silently corrupt clean data, cascading into lower scores at step 10. The agent must triage smartly — not just act.

> **Why this matters:** Bad data is responsible for ~80% of real-world ML project failures. Teaching LLMs to autonomously clean tabular data is a high-value, underexplored frontier.

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🧪 **3 Difficulty Tiers** | Easy / Medium / Hard with distinct corruption profiles and step budgets |
| 🦠 **6 Corruption Classes** | Nulls, type mismatches, duplicates, date format errors, unit mismatches, and adversarial traps |
| 🪤 **Adversarial Traps** | Columns that *look* dirty but are actually clean — penalises trigger-happy agents |
| 📐 **Strict MDP Design** | Stateful sessions, per-step rewards, step budget enforcement, and `done` signals |
| ⚖️ **Tiered Penalty Grader** | Easy=10×, Medium=15×, Hard=20× penalty multiplier for breaking clean data |
| 🔁 **Deterministic Seeding** | Fully reproducible episodes via seed parameter |
| 🧬 **81-Test Suite** | 7 sections covering factory, corruptor, executor, graders, API, E2E, and adversarial cases |
| 🚀 **OpenEnv Compliant** | Native `openenv.yaml`, `/reset`, `/step`, `/health`, `/state` REST API |

---

## 🏗️ Architecture

```
messy-data-triage-gym/
├── inference.py                    # OpenEnv hackathon submission entry point
├── openenv.yaml                    # OpenEnv environment declaration
├── Dockerfile                      # Production container (FastAPI + Uvicorn)
│
├── data_triage_env/
│   ├── server.py                   # FastAPI app: /reset, /step, /health, /state
│   ├── client.py                   # Python client for agent integration
│   ├── models.py                   # Pydantic schemas (Request/Response/Reward)
│   ├── session.py                  # Stateful session manager (UUID-keyed)
│   │
│   ├── engine/
│   │   ├── dataset_factory.py      # Generates clean canonical DataFrames
│   │   ├── corruptor.py            # Injects stochastic corruptions + produces GroundTruthManifest
│   │   └── executor.py             # Applies agent actions and builds observations
│   │
│   └── graders/
│       ├── easy_grader.py          # 10× penalty, no trap cap
│       ├── medium_grader.py        # 15× penalty
│       └── hard_grader.py          # 20× penalty, score capped at 0.5 if trap broken
│
├── baseline.py                     # LLM agent baseline (Llama-3.3-70B via HF Inference)
├── baseline_simple.py              # Heuristic rule-based baseline
└── run_all_tests.py                # 81-test comprehensive validation suite
```

### MDP Definition

| Component | Description |
|---|---|
| **State** | Current DataFrame (shape, dtypes, null counts, head sample, step count) |
| **Action Space** | `inspect`, `cast`, `impute`, `dedupe`, `rescale` |
| **Reward** | Grader score `∈ [0.0, 1.0]` after each action |
| **Terminal Condition** | Score reaches `1.0` OR step budget exhausted |
| **Episode Horizon** | Easy: 20 steps · Medium: 40 steps · Hard: 60 steps |

---

## 🦠 Corruption Taxonomy

| Type | Description | Difficulty |
|---|---|---|
| `null` | NaN injected into numeric/string columns | Easy+ |
| `type_mismatch` | String literals (`"N/A"`, `"unknown"`) replacing numeric cells | Easy+ |
| `duplicate` | Exact row copies appended at end of DataFrame | Medium+ |
| `date_format` | Mixed `DD/MM/YYYY` and `YYYY-MM-DD` in same column | Medium+ |
| `unit_mismatch` | Temperature rows converted C→F with `temp_unit` flipped to `"F"` | Hard |
| `trap` | `pressure` column left clean but resembling "dirty" data — penalises blind rescaling | Hard |

---

## 🎮 Action Space Reference

```python
# 1. Inspect — non-mutating state refresh
{"action": "inspect"}

# 2. Cast — force dtype conversion, with optional regex strip
{"action": "cast", "column": "quantity", "params": {"dtype": "int64"}}
{"action": "cast", "column": "price", "params": {"dtype": "float64", "strip_pattern": "[^\\d.]"}}

# 3. Impute — fill nulls with statistical strategy
{"action": "impute", "column": "price", "params": {"strategy": "median"}}
{"action": "impute", "column": "status", "params": {"strategy": "constant", "value": "unknown"}}

# 4. Dedupe — drop duplicate rows
{"action": "dedupe"}
{"action": "dedupe", "params": {"subset": ["customer_id"], "keep": "first"}}

# 5. Rescale — unit conversion with conditional column filter
{"action": "rescale", "column": "temperature",
 "params": {"from_unit": "F", "to_unit": "C",
            "condition_col": "temp_unit", "condition_val": "F"}}
```

---

## 📊 Difficulty Tiers

### Easy — *The Triage Intern*
- **Dataset:** 200 rows × 4 columns (`customer_id`, `price`, `quantity`, `category`)
- **Corruptions:** 2 — nulls in `price`, type mismatch in `quantity`
- **Budget:** 20 steps
- **Grader:** 10× penalty for touching clean cells

### Medium — *The Junior Analyst*
- **Dataset:** 500 rows × 6 columns (adds `order_date`, `revenue`)
- **Corruptions:** 4 — nulls, type mismatch, duplicate rows, mixed date formats
- **Budget:** 40 steps
- **Grader:** 15× penalty

### Hard — *The Principal Engineer*
- **Dataset:** 800 rows × 8 columns (adds `temperature`, `temp_unit`, `pressure`, `timestamp`)
- **Corruptions:** 6 — all medium corruptions + unit mismatch + adversarial trap + timestamp format
- **Budget:** 60 steps
- **Grader:** 20× penalty · score capped at **0.5** if trap column modified

---

## 📈 Baseline Benchmarks

Results using `meta-llama/Llama-3.3-70B-Instruct` via HF Serverless Inference:

| Task | Score | Steps Used | Notes |
|---|---|---|---|
| Easy | **1.00** | ~6 | Consistently solves nulls + type mismatch |
| Medium | **~0.92** | ~18 | Occasional date format confusion |
| Hard | **~0.78** | ~35 | Trap avoidance is the key differentiator |

> Frontier models (GPT-4o, Claude 3.5) are expected to score higher, especially on Hard.

---

## 🚀 Quickstart

### Local Installation

```bash
git clone <your-repo-url>
cd messy-data-triage-gym

# Install with all extras
pip install -e ".[baseline,test]"
```

### Run the Environment Server

```bash
# Start the FastAPI gym server
uvicorn data_triage_env.server:app --host 0.0.0.0 --port 8000 --reload

# Verify it's healthy
curl http://localhost:8000/health
# {"status": "ok", "env": "messy-data-triage-gym-v1"}

# Browse the interactive API docs
open http://localhost:8000/docs
```

### Docker (Production)

```bash
docker build -t messy-data-triage-gym .
docker run -p 8000:8000 messy-data-triage-gym
```

---

## 🤖 Running Agents

### Heuristic Baseline (No LLM needed)
```bash
python baseline_simple.py
```

### LLM Baseline (Llama-3.3-70B via Hugging Face)
```bash
export HF_TOKEN="hf_..."
python baseline.py
```

### OpenEnv Inference Script
```bash
export HF_TOKEN="hf_..."
export API_BASE_URL="https://api-inference.huggingface.co/v1/"   # default
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"           # default
python inference.py
```

**Expected output format:**
```
[START] task=easy env=messy-data-triage-gym model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action=inspect() reward=0.00 done=false error=null
[STEP] step=2 action=impute('price') reward=0.57 done=false error=null
[STEP] step=3 action=cast('quantity') reward=0.57 done=false error=null
[STEP] step=4 action=impute('quantity') reward=1.00 done=true error=null
[END] success=true steps=4 rewards=0.00,0.57,0.57,1.00
```

---

## 🧪 Testing

A comprehensive **81-test suite** covering all subsystems:

```bash
python run_all_tests.py
```

```
-- SECTION 1: Dataset Factory ----------  10 tests
-- SECTION 2: Corruptor ---------------   9 tests
-- SECTION 3: Executor ----------------  15 tests
-- SECTION 4: Graders -----------------  10 tests
-- SECTION 5: FastAPI Server & Session -  19 tests
-- SECTION 6: Full Episode E2E --------   5 tests
-- SECTION 7: Edge Cases & Adversarial -  13 tests

============================================================
  RESULTS: 81/81 passed  |  0 failed
============================================================
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start new episode (`task_id`, `seed`) |
| `POST` | `/step` | Execute action (`session_id`, `action`) |
| `GET` | `/state/{session_id}` | Inspect current session metadata |
| `GET` | `/docs` | Interactive Swagger UI |

### Reset Request
```json
{"task_id": "hard", "seed": 42}
```

### Step Request
```json
{
  "session_id": "uuid-here",
  "action": {
    "action": "rescale",
    "column": "temperature",
    "params": {"from_unit": "F", "to_unit": "C", "condition_col": "temp_unit", "condition_val": "F"}
  }
}
```

### Step Response
```json
{
  "observation": {
    "shape": [800, 8],
    "columns": [{"name": "temperature", "dtype": "float64", "null_count": 0, ...}],
    "step": 2,
    "message": "Rescaled 160 rows in 'temperature' from F to C"
  },
  "reward": {
    "reward": 0.62,
    "score": 0.62,
    "done": false,
    "info": {"step": 2, "task": "hard", "message": "..."}
  }
}
```

---

## ⚠️ Design Philosophy: Why This Is Hard

1. **Cascading mistakes**: Casting a column to `float64` before imputing prevents `int64` recovery — the agent must plan ahead.
2. **Adversarial traps**: On the Hard tier, blindly rescaling `pressure` (which isn't corrupted) triggers a score cap of 0.5 — the agent must *inspect before acting*.
3. **Irreversible actions**: There is no "undo". A bad `cast` or `rescale` permanently alters the session DataFrame.
4. **Penalty asymmetry**: Breaking a clean cell is penalised 10–20× more severely than leaving a corrupted one un-fixed. Precision matters more than recall.

---

## 📋 OpenEnv Compliance

This environment is fully compliant with the **Meta × Hugging Face OpenEnv RL Challenge** specification:

- ✅ `openenv.yaml` present with correct schema
- ✅ `inference.py` in root with `[START]` / `[STEP]` / `[END]` output format
- ✅ `API_BASE_URL` and `MODEL_NAME` env vars with defaults
- ✅ `HF_TOKEN` validated at startup
- ✅ OpenAI client SDK only (no direct HTTP calls)
- ✅ Docker container runs within 2 vCPU / 8 GB RAM constraints
- ✅ `/health`, `/reset`, `/step` endpoints conformant

---

*Built for the Meta × Hugging Face OpenEnv RL Hackathon.*
