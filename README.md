---
title: Messy Data Triage Gym
emoji: 🧹
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
tags: [openenv]
---

<div align="center">
  <h1>🧹 Messy Data Triage Gym</h1>
  <p><strong>An OpenEnv-compatible Reinforcement Learning environment for autonomous continuous tabular data cleaning.</strong></p>
</div>

---

## 📖 Overview

**Messy Data Triage Gym** is a specialized environment designed for training AI and Reinforcement Learning architectures. It simulates the real-world chaos of data engineering by stochastically injecting dirty, corrupted tabular data, and challenging the agent to systematically sanitize it.

Built structurally compliant with exact OpenEnv RL specifications, the gym teaches autonomous AI frameworks to resolve ubiquitous pipeline failure cases: type mismatching, unrecoverable `nulls`, deduplication, semantic unit transformations, and temporal serialization errors.

## ✨ Features
* **🧪 3 Difficulty Tiers (Easy / Medium / Hard):** Varied row counts and interleaved stochastic corruption levels (e.g. traps and cross-column dependencies).
* **🧠 Deterministic Feedback Grader:** A strict scoring engine evaluating AI actions based on granular dataset permutations and data recovery.
* **🌐 OpenEnv Compliant:** Ready out-of-the-box for AI agent integration with a native `openenv.yaml` schema.
* **⚡ FastAPI Backend:** High-performance REST server for maintaining stateful evaluation sessions.

---

## 📐 Observation & Action Spaces

The environment strictly defines its communication interface using Pydantic schemas.

### Observation Space
The agent receives state updates dynamically formatted as an `Observation` payload:
- **`shape`**: `[rows, columns]` detailing current dimensional state.
- **`columns`**: List of column names and their active datatypes.
- **`missing_counts`**: Dictionary logging `null` distribution.
- **`head`**: A serialized JSON subset showing the first 5 rows of the DataFrame.
- **`message`**: A feedback string representing executor success or HTTP warnings.

### Action Space
Agents manipulate the dataset using strict tool-bound endpoints:
1. `inspect`: Triggers a non-mutating `observation` refresh to evaluate the dataset.
2. `cast`: Forces statistical schema typing (`column`, `params: {dtype: string}`).
3. `impute`: Recovers corrupted structure mathematically (`column`, `params: {strategy: 'median' | 'mean' | 'constant'}`).
4. `dedupe`: Drops colliding row geometries (`params: {keep: 'first'}`).
5. `rescale`: Restores global normalization constants.

---

## 🚀 Quickstart

### Local Installation
Ensure you have Python 3.11+ installed.

```bash
# 1. Clone the repository and install the environment
git clone <your-repo-url>
cd messy-data-triage-gym

# 2. Install via pip including test and baseline dependencies
pip install -e ".[baseline,test]"
```

### Running the API Server

You can run the simulated Gym environment via Uvicorn:
```bash
python -m uvicorn data_triage_env.server:app --host 0.0.0.0 --port 8000 --reload
```
Check `http://localhost:8000/docs` to verify execution.

### Docker Deployment

Production-ready architecture out of the box using multi-stage builds.
```bash
docker build -t messy-data-triage-gym .
docker run -p 8000:8000 messy-data-triage-gym
```

---

## 🤖 Demos & AI Baselines

To visually see an agent interacting, receiving states, and executing tabular data cleaning operations natively against the Gym's API, use our included demos!

### Demonstrated Baselines
Current environment metrics obtained using `meta-llama/Llama-3.3-70B-Instruct` via internal Hugging Face Space tools logic:
* **Easy Task**: 1.00 / 1.0
* **Medium Task**: ~0.92 / 1.0 
* **Hard Task**: ~0.83 / 1.0 *(Difficulty scales based on rigid trap penalties)*

1. **Step-by-step Execution Trace**: 
   Watch a static "perfect" agent navigate the dataset lifecycle.
   ```bash
   python demo.py
   ```

2. **Heuristic Baseline Agent**: 
   Evaluates a rule-based AI script across all three tiers (Easy, Medium, Hard).
   ```bash
   python baseline_simple.py
   ```

3. **Hugging Face Model Baseline (Llama 3.3)**: 
   Runs an autonomous function-calling HF agent against the dynamic Gym parameters. 
   ```bash
   export HF_TOKEN="hf_..."
   python baseline.py
   ```

---

## 🧪 Testing
The platform has been battle-tested against a robust 80+ edge-case test suite covering executor permutations, grading resilience, and FastAPI state boundaries.

To validate your environment:
```bash
python run_all_tests.py
```
