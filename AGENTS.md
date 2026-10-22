# Agent Instructions

## Goal
Maintain a local-first enterprise customer-retention teaching project. Every change should preserve the complete path from ETL to ML to agent evaluation.

## Architecture boundaries
- `pipelines/` owns offline ETL, training, inference, and evaluation.
- `app/repositories/` owns persistence.
- `app/services/` owns business logic.
- `app/tools/` exposes narrow, documented capabilities to agents.
- `app/agents/providers/` is the only layer that may import Microsoft Agent Framework.
- `app/workflows/` coordinates the request lifecycle.
- LLM-generated text must not overwrite authoritative database facts or model outputs.

## Required checks
Run these after behavioral changes:

```bash
ruff check .
pytest -q
python -m scripts.run_evaluation
```

For data/model changes, first run:

```bash
python -m scripts.bootstrap --rows 1200 --max-iter 1000
```

## Safety and quality rules
- Keep CI independent of Ollama and paid cloud APIs.
- Preserve the deterministic fallback.
- Do not add arbitrary SQL execution tools.
- External side effects require an explicit approval boundary.
- Update golden cases or evaluation criteria when intended behavior changes.
- Do not claim causal impact from churn correlations or ROI scenarios.
