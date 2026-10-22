from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.domain import KnowledgeHit


class KnowledgeService:
    def __init__(
        self,
        knowledge_dir: Path,
        min_score: float = 0.05,
        backend: str = "chroma",
        chroma_persist_path: Path | None = None,
        chroma_collection: str = "retention_knowledge",
        embedding_provider: str = "azure_openai",
        azure_openai_endpoint: str | None = None,
        azure_openai_api_key: str | None = None,
        azure_openai_deployment: str | None = None,
        azure_openai_api_version: str = "2024-02-15-preview",
        azure_openai_timeout_seconds: float = 30.0,
    ):
        self.knowledge_dir = knowledge_dir
        self.min_score = min_score
        self.backend = backend.lower().strip()
        self.chroma_persist_path = chroma_persist_path
        self.chroma_collection = chroma_collection
        self.embedding_provider = embedding_provider.lower().strip()
        self.azure_openai_endpoint = (azure_openai_endpoint or "").rstrip("/")
        self.azure_openai_api_key = azure_openai_api_key or ""
        self.azure_openai_deployment = azure_openai_deployment or ""
        self.azure_openai_api_version = azure_openai_api_version
        self.azure_openai_timeout_seconds = azure_openai_timeout_seconds
        self._documents: list[dict[str, str]] = []
        self._doc_ids: list[str] = []
        self._chroma_collection_obj: Any | None = None
        self._chroma_ready = False
        self._chroma_error: str | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if self.backend == "chroma" and self.embedding_provider == "azure_openai":
            if not self._is_azure_embedding_enabled():
                self._chroma_error = "Azure OpenAI embedding configuration is incomplete"

    @property
    def status(self) -> dict[str, str | bool | None]:
        if self.backend != "chroma":
            active_backend = "tfidf"
            degraded = False
        elif self._chroma_ready:
            active_backend = "chroma"
            degraded = False
        elif self._chroma_error:
            active_backend = "tfidf"
            degraded = True
        else:
            active_backend = "pending"
            degraded = False
        return {
            "requested_backend": self.backend,
            "active_backend": active_backend,
            "degraded": degraded,
            "error": self._chroma_error,
        }

    def _is_azure_embedding_enabled(self) -> bool:
        return (
            self.embedding_provider == "azure_openai"
            and bool(self.azure_openai_endpoint)
            and bool(self.azure_openai_api_key)
            and bool(self.azure_openai_deployment)
        )

    def _embed_with_azure_openai(self, texts: list[str]) -> list[list[float]]:
        if not self._is_azure_embedding_enabled():
            raise RuntimeError("Azure OpenAI embedding configuration is incomplete")
        url = (
            f"{self.azure_openai_endpoint}/openai/deployments/{self.azure_openai_deployment}"
            f"/embeddings?api-version={self.azure_openai_api_version}"
        )
        headers = {
            "api-key": self.azure_openai_api_key,
            "Content-Type": "application/json",
        }
        payload = {"input": texts}
        with httpx.Client(timeout=self.azure_openai_timeout_seconds) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        data = response.json().get("data", [])
        if len(data) != len(texts):
            raise RuntimeError("Azure OpenAI embeddings count mismatch")
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [item["embedding"] for item in ordered]

    def _load(self) -> None:
        if self._vectorizer is not None and (
            self.backend != "chroma" or self._chroma_ready or self._chroma_error
        ):
            return
        sections: list[dict[str, str]] = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            current_heading = "Overview"
            buffer: list[str] = []
            for line in text.splitlines():
                if re.match(r"^#{1,3}\s+", line):
                    if buffer:
                        sections.append(
                            {
                                "document": path.name,
                                "section": current_heading,
                                "text": "\n".join(buffer).strip(),
                            }
                        )
                    current_heading = re.sub(r"^#{1,3}\s+", "", line).strip()
                    buffer = []
                elif line.strip():
                    buffer.append(line.strip())
            if buffer:
                sections.append(
                    {
                        "document": path.name,
                        "section": current_heading,
                        "text": "\n".join(buffer).strip(),
                    }
                )
        self._documents = [section for section in sections if section["text"]]
        self._doc_ids = [
            hashlib.sha1(  # noqa: S324
                f"{item['document']}::{item['section']}::{item['text']}".encode()
            ).hexdigest()
            for item in self._documents
        ]
        corpus = [f"{item['section']} {item['text']}" for item in self._documents]
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(corpus) if corpus else None
        self._try_init_chroma()

    def _try_init_chroma(self) -> None:
        if self.backend != "chroma":
            return
        if self._chroma_ready or self._chroma_error:
            return

        try:
            import chromadb

            persist_dir = self.chroma_persist_path or (self.knowledge_dir.parent.parent / "artifacts/chroma")
            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir))
            collection = client.get_or_create_collection(
                name=self.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

            documents = [item["text"] for item in self._documents]
            metadatas = [
                {
                    "document": item["document"],
                    "section": item["section"],
                }
                for item in self._documents
            ]

            if self.embedding_provider == "azure_openai":
                embeddings = self._embed_with_azure_openai(documents)
                collection.upsert(
                    ids=self._doc_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
            else:
                collection.upsert(
                    ids=self._doc_ids,
                    documents=documents,
                    metadatas=metadatas,
                )

            self._chroma_collection_obj = collection
            self._chroma_ready = True
        except Exception as exc:  # noqa: BLE001
            self._chroma_error = str(exc)

    def _search_chroma(self, query: str, top_k: int) -> list[KnowledgeHit]:
        if not self._chroma_ready or self._chroma_collection_obj is None:
            return []
        if self.embedding_provider == "azure_openai":
            query_embedding = self._embed_with_azure_openai([query])[0]
            result = self._chroma_collection_obj.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        else:
            result = self._chroma_collection_obj.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[KnowledgeHit] = []
        for doc_text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            score = max(0.0, 1.0 - float(distance))
            if score < self.min_score:
                continue
            hits.append(
                KnowledgeHit(
                    document=str((metadata or {}).get("document", "knowledge.md")),
                    section=str((metadata or {}).get("section", "Overview")),
                    score=score,
                    text=str(doc_text),
                )
            )
        return hits

    def _search_tfidf(self, query: str, top_k: int) -> list[KnowledgeHit]:
        if not self._documents or self._matrix is None or self._vectorizer is None:
            return []
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        return [
            KnowledgeHit(
                document=self._documents[index]["document"],
                section=self._documents[index]["section"],
                score=float(scores[index]),
                text=self._documents[index]["text"],
            )
            for index in ranked
            if scores[index] >= self.min_score
        ]

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeHit]:
        self._load()
        if self.backend == "chroma":
            hits = self._search_chroma(query, top_k=top_k)
            if hits:
                return hits
        return self._search_tfidf(query, top_k=top_k)
