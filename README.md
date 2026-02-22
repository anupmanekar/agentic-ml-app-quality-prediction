# Agentic Software Quality Prediction

A project demonstrating **MLOps with MLflow** and **Agentic workflows with LangGraph** applied to software defect prediction.

**Stack:** Python · MLflow · LangGraph · LangChain · Claude (Anthropic) · GCP · XGBoost

---

## Problem Statement

Predict software defect probability from code complexity metrics (McCabe, Halstead) and label modules as defect-prone or clean. An agentic system monitors model health, detects data drift, runs retraining experiments, and manages model promotion — all tracked in MLflow.

**Dataset:** NASA JM1 — 10,885 C functions with code metrics and binary defect labels (public, PROMISE repository via OpenML).

---

## Architecture

### ML Layer
- **Problem type:** Binary classification (defect-prone vs clean)
- **Model:** XGBoost with Optuna hyperparameter tuning
- **Features:** McCabe complexity, Halstead metrics, LOC, derived ratio features (26 total)

### MLOps Layer — MLflow
| Concept | Where Used |
|---|---|
| Experiment tracking (`log_params`, `log_metrics`, `log_artifacts`) | All training runs |
| Model flavors (`mlflow.xgboost`) | Model packaging |
| MLflow Model Registry + aliases | `champion` / `challenger` versioning |
| Nested runs | Optuna tuning trials as child runs |
| Model serving (REST endpoint) | `serving/server.py` |
| Programmatic `MlflowClient` API | Promotion script + agent nodes |

### Agentic Layer — LangGraph

**Graph flow:**
```
[drift_monitor]
      ↓
[Drift detected?] ──No──→ [report_writer] → END
      ↓ Yes
[root_cause_analyst]
      ↓
[experiment_designer]
      ↓
[experiment_runner]  ←──────────────────┐
      ↓                                  │
[model_evaluator]                        │
      ↓                                  │
[Evaluation passed?] ──No, retry─────────┘  (max 3 iterations)
      ↓ Yes                              │
      ↓ ──────── Max retries hit ────────→ [report_writer] → END
[human_approval]
      ↓
[Approved?] ──No──→ [report_writer] → END
      ↓ Yes
[registry_manager]
      ↓
[report_writer] → END
```

**Agent nodes:**
| Node | Responsibility | Tools |
|---|---|---|
| `drift_monitor` | Compute PSI/KS stats vs training baseline | pandas, scipy |
| `root_cause_analyst` | Reason over drift signals, identify feature shifts | Claude + MLflow client |
| `experiment_designer` | Propose hyperparameter / feature changes | Claude |
| `experiment_runner` | Train challenger model, log to MLflow | MLflow client |
| `model_evaluator` | Compare challenger vs champion metrics | MLflow client |
| `human_approval` | Pause graph, await human decision | LangGraph interrupt |
| `registry_manager` | Set `champion` alias in MLflow registry | MLflow client |
| `report_writer` | Summarise retraining cycle | Claude |

---

## Project Structure

```
mlops-with-mlflow/
├── README.md
├── pyproject.toml                      # uv project dependencies
├── .env.example                        # Environment variable template
├── .env                                # Your local config (gitignored)
│
├── configs/
│   ├── model_config.yaml               # Thresholds, hyperparams, experiment name
│   ├── best_params.yaml                # Best params from Optuna tuning (generated)
│   └── agent_config.yaml               # Agent model, drift config, max iterations
│
├── data/
│   ├── raw/                            # Source datasets (gitignored)
│   ├── processed/                      # Engineered features (gitignored)
│   └── drift_samples/                  # New data for drift detection (gitignored)
│
├── training/
│   ├── features/feature_engineering.py # Clean, encode, derive features
│   ├── models/xgboost_model.py         # Model builder + cross-validation
│   ├── pipelines/
│   │   ├── train.py                    # Main training pipeline
│   │   ├── tune.py                     # Optuna tuning with MLflow nested runs
│   │   └── promote.py                  # Model Registry promotion script
│   └── evaluation/evaluator.py         # Metrics + plots (shared with agent)
│
├── agents/
│   ├── state.py                        # AgentState TypedDict (shared state schema)
│   ├── graph.py                        # Graph wiring + routing functions
│   ├── nodes/                          # One file per agent node (Phase 3)
│   │   ├── drift_monitor.py
│   │   ├── root_cause_analyst.py
│   │   ├── experiment_designer.py
│   │   ├── experiment_runner.py
│   │   ├── model_evaluator.py
│   │   ├── human_approval.py
│   │   ├── registry_manager.py
│   │   └── report_writer.py
│   └── tools/
│       ├── mlflow_tools.py             # LangChain tools wrapping MLflow client
│       └── stats_tools.py             # PSI / KS test utilities
│
├── serving/
│   ├── server.py                       # Start MLflow model serving endpoint
│   └── predict.py                      # REST prediction client
│
├── infra/
│   ├── docker/Dockerfile.mlflow        # Phase 5 — GCP deployment
│   └── scripts/
│       ├── start_mlflow_local.sh       # Start local MLflow tracking server
│       ├── setup_mlflow_experiment.py  # Create MLflow experiment (run once)
│       ├── download_data.py            # Download NASA JM1 dataset
│       └── smoke_test.py               # Verify MLflow tracking end-to-end
│
├── notebooks/                          # EDA and baseline experimentation
└── tests/
    ├── unit/
    └── integration/
```

---

## Local Development Setup

Complete these steps in order when setting up for the first time.

### Prerequisites
- Python 3.11+
- `uv` package manager

### Step 1 — Create virtual environment
```bash
uv venv --python 3.11
```

### Step 2 — Install dependencies
```bash
uv sync --dev
```

### Step 3 — Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and fill in your `ANTHROPIC_API_KEY`. GCP values are only needed for Phase 5.

### Step 4 — Start the MLflow tracking server
Run this in a dedicated terminal and keep it running:
```bash
uv run bash infra/scripts/start_mlflow_local.sh
```

| Config | Value |
|---|---|
| Backend store | `mlruns/mlflow.db` (SQLite) |
| Artifact root | `mlartifacts/` |
| UI | http://localhost:5000 |

### Step 5 — Create the MLflow experiment
Run once after the server is up:
```bash
uv run python infra/scripts/setup_mlflow_experiment.py
```

### Step 6 — Verify with smoke test
```bash
uv run python infra/scripts/smoke_test.py
```

---

## Phase 1 — Training Pipeline

```bash
# Download dataset
uv run python infra/scripts/download_data.py

# Baseline training run
uv run python -m training.pipelines.train --run-name baseline-xgboost

# Hyperparameter tuning (30 Optuna trials logged as nested MLflow child runs)
uv run python -m training.pipelines.tune --n-trials 30

# Final training with best params from tuning
uv run python -m training.pipelines.train \
    --config configs/best_params.yaml \
    --run-name tuned-final
```

---

## Phase 2 — Model Registry & Serving

```bash
# Preview promotion without making changes
uv run python -m training.pipelines.promote --dry-run

# Promote best model as champion (auto-selects by AUC-ROC)
uv run python -m training.pipelines.promote

# Promote a specific version
uv run python -m training.pipelines.promote --version 6

# Start model serving endpoint (dedicated terminal — keep running)
# Requires: MLflow tracking server running + champion alias set
uv run python -m serving.server

# Score samples
uv run python -m serving.predict
uv run python -m serving.predict --from-test --rows 10
```

---

## Phase 3 — LangGraph Agent System *(in progress)*

The agent system automates the full retraining cycle:
drift detection → root cause analysis → experiment design → training → evaluation → human approval → promotion.

---

## Subsequent Sessions

Steps 1–3 are one-time only. From the second session onwards:

```bash
# Terminal 1 — MLflow tracking server (keep running)
uv run bash infra/scripts/start_mlflow_local.sh

# Terminal 2 — Model serving (keep running, after champion is set)
uv run python -m serving.server

# Terminal 3 — your working terminal
uv run python -m training.pipelines.train --run-name my-run
```

---

## GCP Deployment — Phase 5

| Component | GCP Service |
|---|---|
| MLflow tracking server | Cloud Run |
| MLflow artifact store | GCS bucket |
| MLflow backend store | Cloud SQL (PostgreSQL) |
| Model serving endpoint | Cloud Run |
| Pipeline scheduling | Cloud Scheduler + Cloud Run Jobs |
| Container registry | Artifact Registry |
