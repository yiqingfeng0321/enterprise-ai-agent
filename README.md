# Enterprise Customer Retention Agent

A **local-first enterprise AI teaching system** that combines ETL, supervised machine learning, MLflow, Microsoft Agent Framework, Ollama, memory, RAG, evaluation, FastAPI, Streamlit, Docker, GitHub Actions, and an Azure migration path.

The default teaching scenario is **B2B AI copilot subscription retention**: think Microsoft 365 Copilot-style enterprise accounts where adoption, champions, support friction, renewal timing, and seat utilization all influence churn risk.

The student-safe default uses the deterministic provider, repository-local Agent Skills,
and lexical TF-IDF retrieval, so the complete workflow runs without cloud credentials or
Ollama. Microsoft Agent Framework with Ollama is the next-stage local LLM option. Chroma
with Azure OpenAI embeddings is opt-in and its fallback status is exposed by `/health`.

## What actually runs

```text
Historical labeled copilot accounts + current active accounts
        ↓
ETL + schema/data-quality validation
        ↓
SQLite history/current feature tables
        ↓
Churn model trained on historical accounts + MLflow-compatible tracking
        ↓
Current-customer scoring + counterfactual feature explanations
        ↓
ML / SQL / Knowledge / Memory sub-agents
        ↓
Recommendation + reviewer + report agents
        ↓
FastAPI + Streamlit + golden-dataset evaluation
```

## Quick start

New students should follow the complete, platform-specific
[Student Operation Manual (PDF)](docs/学生操作手册_JohnDataLab_2026.pdf). Students can
either clone from GitHub or use a complete ZIP package; the deterministic first run
does not require Ollama, Docker, Azure, or paid APIs.

```bash
python3.11 -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set AGENT_PROVIDER=deterministic
python -m scripts.bootstrap
python -m scripts.demo
```

The repository targets CPython 3.11 and uses `constraints.txt` to pin the resolved
environment. Ollama is an optional second-stage exercise.

`python -m scripts.bootstrap` now generates two raw files:

- `data/raw/customer_history.csv` for supervised training with churn labels.
- `data/raw/current_customers.csv` for live scoring without churn labels.

Start the API:

```bash
uvicorn app.api.main:app --reload
```

Start the UI in another terminal:

```bash
streamlit run app/ui/streamlit_app.py
```

Run tests and evaluation:

```bash
pytest -q
python -m scripts.run_evaluation
```

`make test-all` runs the complete deterministic API/workflow/golden path without Ollama. Evaluation is read-only with respect to operational predictions and memory, records provider provenance, and exits non-zero when a quality threshold is missed.

## MLflow tracking and hyperparameter tuning

Start MLflow locally:

```bash
docker compose up -d mlflow
```

> Ports: `docker compose` publishes MLflow on host port **5001** because macOS reserves port 5000
> for the AirPlay Receiver (requests to `localhost:5000` reach AirPlay and fail with 403). If you
> run `mlflow server` yourself, pick a free port and use `127.0.0.1` rather than `localhost`.
> XGBoost on macOS also needs OpenMP: `brew install libomp`.

Enable tracking in your shell or `.env`:

```dotenv
ENABLE_MLFLOW=true
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
MLFLOW_EXPERIMENT_NAME=enterprise-retention-agent
MLFLOW_USE_MODEL_REGISTRY=true
MLFLOW_REGISTERED_MODEL_NAME=b2b-ai-copilot-churn-risk-model
MLFLOW_DEPLOYMENT_ALIAS=champion
```

Run standard training with MLflow logging:

```bash
python -m scripts.train_model
```

Register the newly trained model and move the deployed alias:

```bash
python -m scripts.train_model --register-model
```

Run a grid search and promote the best model:

```bash
python -m scripts.tune_hyperparams \
        --models logistic_regression,random_forest,xgboost,deep_learning \
        --logistic-max-iter 300,600 \
        --logistic-c 0.3,0.5 \
        --rf-n-estimators 200,400 \
        --rf-max-depth 8,12 \
        --xgb-n-estimators 200,400 \
        --xgb-max-depth 4,6 \
        --dl-hidden-layers 64x32,128x64 \
        --metric roc_auc \
        --enable-mlflow \
        --promote-best \
        --register-best
```

## MLflow Model Registry and rollback

The project can use MLflow Model Registry as the deployment source of truth.
When `MLFLOW_USE_MODEL_REGISTRY=true`, the runtime loads the churn model from:

```text
models:/b2b-ai-copilot-churn-risk-model@champion
```

Recommended operating pattern:

1. Train or tune a candidate model with MLflow enabled.
2. Register the chosen run and move the `champion` alias.
3. Let the API/UI load the deployed alias instead of a hand-copied local file.
4. If needed, roll back `champion` to `previous_champion` or a specific version.

Rollback examples:

```bash
python -m scripts.rollback_model
python -m scripts.rollback_model --version 3
```

By default, rollback also syncs the selected registry version back into the local
`MODEL_PATH` and `MODEL_METADATA_PATH` files, so non-registry local runs still match
the deployed alias.

Open MLflow UI:

```text
http://127.0.0.1:5001
```

## Use Microsoft Agent Framework + Ollama

Install Ollama and pull a tool-capable model:

```bash
ollama pull qwen2.5:7b
pip install --pre -c constraints.txt -e ".[dev,ui]"
```

Update `.env`:

```dotenv
AGENT_PROVIDER=microsoft_ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
ALLOW_DETERMINISTIC_FALLBACK=false
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1500
LLM_TIMEOUT_SECONDS=120
LLM_PROMPT_MODE=lean
LLM_NUM_CTX=8192
```

`LLM_PROMPT_MODE` controls how much the local model is asked to carry at run time:

- `lean` (default): the workflow has already fetched the profile, model output, playbooks and
  memory, so the prompt only carries the compact facts, the skills chosen by the application-side
  router, and the JSON schema. Tool and skill definitions are not sent to the model, JSON mode is
  on, and a typical analysis takes about a minute on a 7B model.
- `full`: sends the complete context and exposes all tools plus `load_skill` /
  `read_skill_resource` for progressive disclosure. This is the industrial-style path; it is
  several times slower on small local models and can exceed the timeout.

`LLM_NUM_CTX` sets Ollama's context window. Ollama's default (4096) silently truncates long
prompts, which is why the default path was falling back to the deterministic provider.

The provider layer remains modular:

- `app/agents/providers/microsoft_ollama.py` (default; Microsoft Agent Framework)
- `app/agents/providers/langgraph_ollama.py` (alternative orchestration path)
- `app/agents/providers/deterministic.py`

Formal skills live under `skills/<skill-name>/SKILL.md`. Microsoft Agent Framework advertises
their metadata and exposes `load_skill` and `read_skill_resource` for progressive disclosure.
The LangGraph alternative uses the same repository loader and application-side skill router.
Internal people roles remain separate in `app/services/specialist_catalog.py`.

Switch providers only when needed for diagnostics or A/B behavior checks. CI explicitly uses
the deterministic provider, so tests do not require Ollama or a paid API.

## Use Azure OpenAI for cloud recommendation generation

The recommendation path can switch from local Ollama to Azure OpenAI without changing the
rest of the app. The code uses raw HTTP calls, so no Azure SDK is required in the teaching
repo.

Add this to `.env` or supply the same values through your Azure deployment:

```dotenv
AGENT_PROVIDER=azure_openai
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=<your-chat-deployment>
AZURE_OPENAI_API_VERSION=2024-06-01
AZURE_OPENAI_TIMEOUT_SECONDS=120
ALLOW_DETERMINISTIC_FALLBACK=true
```

The Azure deployment template in `infra/azure/main.bicep` wires these settings into the
container app so the same code can run locally or in Azure.

## Use Azure OpenAI embeddings for RAG retrieval

The knowledge retrieval path supports a real vector database backend using Chroma with
Azure OpenAI embeddings. This avoids local default embedding model downloads and keeps
embedding behavior consistent across environments.

Add the following to `.env`:

```dotenv
RETRIEVAL_BACKEND=chroma
RETRIEVAL_EMBEDDING_PROVIDER=azure_openai
RETRIEVAL_CHROMA_PERSIST_PATH=artifacts/chroma
RETRIEVAL_CHROMA_COLLECTION=retention_knowledge
RETRIEVAL_AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
RETRIEVAL_AZURE_OPENAI_API_KEY=<your-api-key>
RETRIEVAL_AZURE_OPENAI_DEPLOYMENT=text-embedding-3-large
RETRIEVAL_AZURE_OPENAI_API_VERSION=2024-02-15-preview
RETRIEVAL_AZURE_OPENAI_TIMEOUT_SECONDS=30
```

Install dependencies:

```bash
pip install -r requirements.txt
```

When Chroma cannot initialize, retrieval falls back to TF-IDF and `/health` reports `retrieval_degraded=true` with the reason.

## Main endpoints

- `GET /health`
- `GET /v1/customers/{customer_id}`
- `GET /v1/customers/high-risk?limit=10`
- `POST /v1/analysis`
- `POST /v1/feedback`

Example:

```bash
curl -X POST http://localhost:8000/v1/analysis \
  -H 'Content-Type: application/json' \
  -d '{"customer_id":"CUST-0001","session_id":"demo","create_ticket":false}'
```

## Local Airflow cloud-endpoint demo

The Airflow demo simulates a production schedule without requiring a cloud account. A
local FastAPI service stands in for both the upstream customer-data endpoint and the
downstream prediction-publication endpoint:

```text
Mock cloud GET endpoint
        ↓
Airflow extract → ETL → batch inference
        ↓
high_risk_customers.csv
        ↓
Mock cloud POST endpoint
```

Prepare the model and database once, then start the demo:

```bash
python -m scripts.bootstrap --rows 1200 --max-iter 1000
docker compose -f docker-compose.airflow.yml up --build
```

Open `http://localhost:8080`, select `daily_customer_risk_refresh`, and click
**Trigger DAG**. This local-only configuration uses Airflow's simple authentication
manager with all-admin mode, so it does not ask for a password. The mock cloud API is
available at `http://localhost:9000/docs`.

After a successful run:

- `artifacts/predictions.csv` contains all refreshed scores.
- `artifacts/high_risk_customers.csv` contains the high-risk subset.
- `artifacts/cloud_demo/latest_predictions.json` shows what was sent to the simulated
  cloud endpoint.

The DAG is scheduled for 06:00 America/Los_Angeles and does not backfill missed dates.
Publishing is protected by the Airflow variable
`customer_risk_publish_enabled`. The local Compose demo sets it to `true` because its
target is the local mock server. Keep it `false` until publication to a real endpoint has
been explicitly approved.

To point the same DAG at a real endpoint, replace
`AIRFLOW_CONN_CUSTOMER_CLOUD_API` in `docker-compose.airflow.yml` with an Airflow secret
or connection whose host is the real base URL and whose password is the API key. Do not
commit a real credential to the Compose file.

Stop the demo with:

```bash
docker compose -f docker-compose.airflow.yml down
```

## Repository layering

```text
API / Frontend
    ↓
Business Workflow
    ↓
Agents
    ↓
Tools
    ↓
Services
    ↓
Repositories
    ↓
SQLite / model artifact / knowledge documents
```

See `docs/core/architecture.md` and `docs/core/prd.md` for implementation details.

For a guided explanation of the repository (ETL, ML, agents, multi-agent, Docker, Airflow),
read the [Code Walkthrough Handbook (PDF)](docs/代码讲解手册_JohnDataLab_2026.pdf).
