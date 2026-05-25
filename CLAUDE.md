# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is TaskRouter

Adaptive LLM routing engine that decides whether to send tasks to a local Ollama model (free) or a cloud API (paid) using Thompson Sampling Bayesian decision-making. Every routing decision feeds back into the system for online learning. Version 6.0.0.

## Commands

```bash
# Install (editable, includes dev deps)
pip install -e ".[dev]"

# Tests (700 tests, all must pass)
python3 -m pytest tests/ -v
python3 -m pytest tests/ -v --cov=src/task_router --cov-report=term-missing  # with coverage
python3 -m pytest tests/test_tqbc.py -v          # single file
python3 -m pytest tests/test_tqbc.py -v -k "test_name"  # single test

# Lint (CI enforces these)
ruff check src/task_router/ --select E,F,W --ignore E501,E402,F541
ruff format src/task_router/ --check --diff

# Benchmarks (standalone, not part of pytest)
python3 benchmark_tqbc.py
python3 benchmark_learning_curve.py

# Run the server
sma serve                          # CLI entry point
python3 scripts/api_server.py      # direct
docker build -t taskrouter . && docker run -p 8930:8930 taskrouter
```

## Architecture

### Dual Source Trees

Two copies of the codebase exist:

- **`src/task_router/`** (42 files) — The installable Python package. This is the canonical source. Uses relative imports.
- **`scripts/`** (36 files) — Flat standalone copies used by the Dockerfile and `install.sh`. Missing 6 modules that only exist in `src/`: `episode_collector.py`, `quality_judge.py`, `reflection.py`, `correction_applier.py`, `cost.py`, `preprocessing.py`.

**When editing, always modify files in `src/task_router/`**, not `scripts/`.

### Core Request Flow

```
Input → SemanticCache (3-tier: exact/normalized/fuzzy Jaccard)
  Cache miss →
    A3M complexity scoring (routing.py)
    → TQBCRouter (tqbc.py) — Thompson Sampling Bayesian cascade
    → ConformalizedRouter (conformal_routing.py) — uncertainty quantification
    → MetaLearner (meta_learner.py) — online logistic regression fusion
    → Reasoning strategy selection (reasoning.py) — 5 strategies
    → Adaptive compression (adaptive_compression.py)
    → Model execution — Ollama local or cloud API (models.py)
    → Output validation (validation.py)
    → Cloud fallback if local quality is poor
  → Distillation pair collection
  → Outcome-aware cache update (outcome_cache.py)
  → Audit logging (audit.py)
  → Episode collection → Self-Reflective Reflection (SRR)
```

### Key Modules

| Module | Role |
|--------|------|
| `task_router.py` | Central orchestrator — wires everything together |
| `tqbc.py` | Core routing engine. Token quantile features (q25/q50/q75/q90) + Thompson Sampling |
| `conformal_routing.py` | Adaptive Conformal Inference — distribution-free coverage guarantees |
| `meta_learner.py` | 10-dim feature fusion via online logistic regression (<100 params) |
| `routing.py` | `Task` dataclass, A3M complexity scoring, task decomposition |
| `confidence.py` | Confidence extraction from token logprobs, gating cascade |
| `models.py` | `call_ollama()` / `call_cloud_api()` with CircuitBreaker |
| `cache.py` | Semantic cache with trigram Jaccard fuzzy matching |
| `reasoning.py` | Strategy selector: direct, cot, cod, few_shot, structured |
| `config.py` | `RouterConfig` dataclass, YAML/JSON/env loading |
| `privacy.py` | PII detection (Chinese IDs, phones, etc.) — applied only to cloud calls |

### Self-Reflective Routing (SRR) — v6.0

Only in `src/task_router/`, not in `scripts/`:
- `episode_collector.py` — Snapshots after each `run_task()` call
- `quality_judge.py` — LLM-as-Judge (5 dimensions)
- `reflection.py` — Three-layer reflection: RouteAnalyzer → StrategyReflector → JointReflector
- `correction_applier.py` — Applies corrections with safety bounds (confidence ≥ 0.6, ±0.5 threshold limit, rollback support)

## Configuration

- Config file: `config.example.yaml` → copy to `config.yaml`
- Env vars: `TASKROUTER_LOCAL_MODEL`, `TASKROUTER_CLOUD_API_URL`, `TASKROUTER_CLOUD_API_KEY`, `TASK_ROUTER_CACHE`
- Persistent state (cache, distillation pairs, logs) stored in `~/.cache/task_router/` as JSONL files
- Thread safety: all shared state uses `threading.Lock`

## CI

GitHub Actions runs on push/PR to `main`: Python 3.10–3.13 matrix, pytest with ≥40% coverage, ruff lint+format check, py_compile syntax check.
