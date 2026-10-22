from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    database_path: Path
    model_path: Path
    model_metadata_path: Path
    raw_history_data_path: Path
    raw_current_data_path: Path
    knowledge_dir: Path
    golden_dataset_path: Path
    skills_dir: Path
    skill_max_resource_bytes: int
    agent_provider: str
    ollama_model: str
    ollama_base_url: str
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_deployment: str
    azure_openai_api_version: str
    azure_openai_timeout_seconds: float
    allow_deterministic_fallback: bool
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int
    llm_prompt_mode: str
    llm_num_ctx: int
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    enable_mlflow: bool
    mlflow_use_model_registry: bool
    mlflow_registered_model_name: str
    mlflow_deployment_alias: str
    log_level: str
    max_review_iterations: int
    risk_high: float
    risk_medium: float
    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_backend: str
    retrieval_chroma_persist_path: Path
    retrieval_chroma_collection: str
    retrieval_embedding_provider: str
    retrieval_azure_openai_endpoint: str
    retrieval_azure_openai_api_key: str
    retrieval_azure_openai_deployment: str
    retrieval_azure_openai_api_version: str
    retrieval_azure_openai_timeout_seconds: float
    max_session_messages: int
    max_customer_memories: int

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_history_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_current_data_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.golden_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.retrieval_chroma_persist_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    environment = os.getenv("APP_ENV", "local")
    config = _deep_merge(
        _load_yaml(PROJECT_ROOT / "configs/base.yaml"),
        _load_yaml(PROJECT_ROOT / f"configs/{environment}.yaml"),
    )
    storage = config.get("storage", {})
    agent = config.get("agent", {})
    retrieval = config.get("retrieval", {})
    memory = config.get("memory", {})
    skills = config.get("skills", {})
    application = config.get("application", {})

    settings = Settings(
        app_name=os.getenv("APP_NAME", application.get("name", "Enterprise Retention Agent")),
        environment=environment,
        database_path=_absolute(os.getenv("DATABASE_PATH", storage.get("database_path", "artifacts/retention.db"))),
        model_path=_absolute(os.getenv("MODEL_PATH", storage.get("model_path", "artifacts/models/churn_model.joblib"))),
        model_metadata_path=_absolute(
            os.getenv(
                "MODEL_METADATA_PATH",
                storage.get("model_metadata_path", "artifacts/models/churn_model_metadata.json"),
            )
        ),
        raw_history_data_path=_absolute(
            os.getenv(
                "RAW_HISTORY_DATA_PATH",
                storage.get("raw_history_data_path", "data/raw/customer_history.csv"),
            )
        ),
        raw_current_data_path=_absolute(
            os.getenv(
                "RAW_CURRENT_DATA_PATH",
                storage.get("raw_current_data_path", "data/raw/current_customers.csv"),
            )
        ),
        knowledge_dir=_absolute(os.getenv("KNOWLEDGE_DIR", storage.get("knowledge_dir", "data/knowledge"))),
        golden_dataset_path=_absolute(
            os.getenv(
                "GOLDEN_DATASET_PATH",
                storage.get("golden_dataset_path", "data/golden/customer_retention_cases.jsonl"),
            )
        ),
        skills_dir=_absolute(os.getenv("AGENT_SKILLS_DIR", skills.get("root_path", "skills"))),
        skill_max_resource_bytes=int(
            os.getenv(
                "AGENT_SKILL_MAX_RESOURCE_BYTES",
                skills.get("max_resource_bytes", 100_000),
            )
        ),
        agent_provider=os.getenv("AGENT_PROVIDER", agent.get("provider", "microsoft_ollama")),
        ollama_model=os.getenv("OLLAMA_MODEL", agent.get("ollama_model", "qwen2.5:7b")),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            os.getenv("OLLAMA_HOST", agent.get("ollama_base_url", "http://localhost:11434")),
        ),
        azure_openai_endpoint=os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            agent.get("azure_openai_endpoint", ""),
        ),
        azure_openai_api_key=os.getenv(
            "AZURE_OPENAI_API_KEY",
            agent.get("azure_openai_api_key", ""),
        ),
        azure_openai_deployment=os.getenv(
            "AZURE_OPENAI_DEPLOYMENT",
            agent.get("azure_openai_deployment", ""),
        ),
        azure_openai_api_version=os.getenv(
            "AZURE_OPENAI_API_VERSION",
            agent.get("azure_openai_api_version", "2024-06-01"),
        ),
        azure_openai_timeout_seconds=float(
            os.getenv(
                "AZURE_OPENAI_TIMEOUT_SECONDS",
                agent.get("azure_openai_timeout_seconds", 120.0),
            )
        ),
        allow_deterministic_fallback=_coerce_bool(
            os.getenv("ALLOW_DETERMINISTIC_FALLBACK", agent.get("allow_deterministic_fallback", "false")),
            default=False,
        ),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", agent.get("llm_temperature", 0.2))),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", agent.get("llm_max_tokens", 1500))),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", agent.get("llm_timeout_seconds", 120))),
        llm_prompt_mode=os.getenv("LLM_PROMPT_MODE", agent.get("llm_prompt_mode", "lean")).strip().lower(),
        llm_num_ctx=int(os.getenv("LLM_NUM_CTX", agent.get("llm_num_ctx", 8192))),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "enterprise-retention-agent"),
        enable_mlflow=_coerce_bool(os.getenv("ENABLE_MLFLOW", "false"), default=False),
        mlflow_use_model_registry=_coerce_bool(
            os.getenv("MLFLOW_USE_MODEL_REGISTRY", "false"),
            default=False,
        ),
        mlflow_registered_model_name=os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            "b2b-ai-copilot-churn-risk-model",
        ),
        mlflow_deployment_alias=os.getenv(
            "MLFLOW_DEPLOYMENT_ALIAS",
            "champion",
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_review_iterations=int(agent.get("max_review_iterations", 1)),
        risk_high=float(agent.get("risk_thresholds", {}).get("high", 0.70)),
        risk_medium=float(agent.get("risk_thresholds", {}).get("medium", 0.40)),
        retrieval_top_k=int(retrieval.get("top_k", 3)),
        retrieval_min_score=float(retrieval.get("min_score", 0.05)),
        retrieval_backend=os.getenv("RETRIEVAL_BACKEND", retrieval.get("backend", "lexical")),
        retrieval_chroma_persist_path=_absolute(
            os.getenv(
                "RETRIEVAL_CHROMA_PERSIST_PATH",
                retrieval.get("chroma_persist_path", "artifacts/chroma"),
            )
        ),
        retrieval_chroma_collection=os.getenv(
            "RETRIEVAL_CHROMA_COLLECTION",
            retrieval.get("chroma_collection", "retention_knowledge"),
        ),
        retrieval_embedding_provider=os.getenv(
            "RETRIEVAL_EMBEDDING_PROVIDER",
            retrieval.get("embedding_provider", "azure_openai"),
        ),
        retrieval_azure_openai_endpoint=os.getenv(
            "RETRIEVAL_AZURE_OPENAI_ENDPOINT",
            retrieval.get("azure_openai_endpoint", ""),
        ),
        retrieval_azure_openai_api_key=os.getenv(
            "RETRIEVAL_AZURE_OPENAI_API_KEY",
            retrieval.get("azure_openai_api_key", ""),
        ),
        retrieval_azure_openai_deployment=os.getenv(
            "RETRIEVAL_AZURE_OPENAI_DEPLOYMENT",
            retrieval.get("azure_openai_deployment", ""),
        ),
        retrieval_azure_openai_api_version=os.getenv(
            "RETRIEVAL_AZURE_OPENAI_API_VERSION",
            retrieval.get("azure_openai_api_version", "2024-02-15-preview"),
        ),
        retrieval_azure_openai_timeout_seconds=float(
            os.getenv(
                "RETRIEVAL_AZURE_OPENAI_TIMEOUT_SECONDS",
                retrieval.get("azure_openai_timeout_seconds", 30.0),
            )
        ),
        max_session_messages=int(memory.get("max_session_messages", 12)),
        max_customer_memories=int(memory.get("max_customer_memories", 8)),
    )
    if not 0 <= settings.risk_medium < settings.risk_high <= 1:
        raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")
    settings.ensure_directories()
    return settings
