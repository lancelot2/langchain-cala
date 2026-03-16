"""
CalaRetriever — LangChain BaseRetriever wrapper for the Cala knowledge API.

Cala (https://cala.ai) is a verified entity graph for AI agents.
It returns structured, typed, traceable data instead of raw web scrapes.

Usage::

    from langchain_cala import CalaRetriever

    retriever = CalaRetriever(api_key="YOUR_CALA_API_KEY")
    docs = retriever.invoke("Latest funding rounds for Barcelona AI startups")
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import requests
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, SecretStr, model_validator


# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.cala.ai"
DEFAULT_TIMEOUT = 30  # seconds


# ── Main Retriever ───────────────────────────────────────────────────────────


class CalaRetriever(BaseRetriever):
    """LangChain retriever backed by the Cala verified knowledge graph API.

    Cala returns structured, typed, source-cited data — not raw web results.
    Every document returned includes ``source`` and ``verified`` metadata
    fields so your agent can reason over facts with full traceability.

    Modes
    -----
    ``"search"`` *(default)*
        Natural-language question → LLM-synthesised answer with citations.
        Use this for open-ended questions: *"Who funds Barcelona AI startups?"*

    ``"query"``
        Structured dot-notation query → typed field value.
        Use this for precise lookups: *"OpenAI.founded.year"*

    ``"entities"``
        Entity discovery by name → list of matching entities with metadata.
        Use this to ground a name to a known entity before deeper queries.

    Parameters
    ----------
    api_key:
        Your Cala API key. Falls back to the ``CALA_API_KEY`` env variable.
    mode:
        One of ``"search"`` | ``"query"`` | ``"entities"``. Default: ``"search"``.
    k:
        Maximum number of documents to return. Default: 5.
    timeout:
        HTTP request timeout in seconds. Default: 30.

    Examples
    --------
    Basic search::

        retriever = CalaRetriever(api_key="...")
        docs = retriever.invoke("AI regulation in the EU")

    Structured query::

        retriever = CalaRetriever(api_key="...", mode="query")
        docs = retriever.invoke("Mistral.founded.year")

    Entity discovery::

        retriever = CalaRetriever(api_key="...", mode="entities")
        docs = retriever.invoke("Factorial HR")

    As a chain component::

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_template(
            "Answer based on the context below.\\n\\nContext:\\n{context}\\n\\nQuestion: {question}"
        )
        chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | ChatOpenAI()
            | StrOutputParser()
        )
        chain.invoke("Which EU AI startups raised funding in 2024?")
    """

    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("CALA_API_KEY", "")),
        description="Cala API key. Falls back to CALA_API_KEY env variable.",
    )
    mode: Literal["search", "query", "entities"] = Field(
        default="search",
        description="API mode: 'search' (NL), 'query' (structured), 'entities' (discovery).",
    )
    k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of documents to return.",
    )
    timeout: int = Field(
        default=DEFAULT_TIMEOUT,
        description="HTTP request timeout in seconds.",
    )

    # ── Validation ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_api_key(self) -> "CalaRetriever":
        if not self.api_key.get_secret_value():
            raise ValueError(
                "No Cala API key found. Pass api_key= or set the CALA_API_KEY "
                "environment variable. Get a key at https://console.cala.ai"
            )
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-KEY": self.api_key.get_secret_value(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _make_request(
        self, method: str, path: str, **kwargs: Any
    ) -> Dict[str, Any]:
        url = f"{BASE_URL}{path}"
        try:
            response = requests.request(
                method,
                url,
                headers=self._headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Cala API timed out after {self.timeout}s. "
                "Increase timeout= if needed."
            )
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            body = ""
            if e.response is not None:
                try:
                    body = e.response.json().get("message", e.response.text)
                except Exception:
                    body = e.response.text
            raise RuntimeError(
                f"Cala API error {status}: {body}"
            ) from e
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cala API request failed: {e}") from e

    # ── Mode handlers ─────────────────────────────────────────────────────────

    def _search(self, query: str) -> List[Document]:
        """POST /v1/knowledge/search — NL question, returns answer + sources."""
        data = self._make_request(
            "POST",
            "/v1/knowledge/search",
            json={"input": query},
        )

        docs: List[Document] = []

        # Primary answer document — field is "content" per API spec
        content = data.get("content", "")
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "cala:knowledge_search",
                        "query": query,
                        "verified": True,
                        "type": "answer",
                        # Include reasoning steps if present
                        "explainability": data.get("explainability", []),
                    },
                )
            )

        # Supporting facts — field is "context" (list of KnowBit objects)
        context_facts = data.get("context") or []
        for fact in context_facts[: self.k]:
            fact_content = (
                fact.get("text")
                or fact.get("content")
                or fact.get("value")
                or str(fact)
            )
            docs.append(
                Document(
                    page_content=fact_content,
                    metadata={
                        "source": fact.get("source") or fact.get("url", "cala"),
                        "verified": fact.get("verified", True),
                        "type": "context_fact",
                        "query": query,
                        "entity": fact.get("entity"),
                        "field": fact.get("field"),
                    },
                )
            )

        # Entity mentions — field is "entities" (list of EntityMention objects)
        entities = data.get("entities") or []
        for ent in entities[: self.k]:
            ent_content = (
                ent.get("description")
                or ent.get("summary")
                or ent.get("name", "")
            )
            if ent_content:
                docs.append(
                    Document(
                        page_content=ent_content,
                        metadata={
                            "source": "cala:entity",
                            "entity_id": ent.get("id"),
                            "entity_name": ent.get("name"),
                            "entity_type": ent.get("type"),
                            "verified": True,
                            "type": "entity_mention",
                            "query": query,
                        },
                    )
                )

        return docs

    def _query(self, query: str) -> List[Document]:
        """POST /v1/knowledge/query — structured dot-notation query."""
        data = self._make_request(
            "POST",
            "/v1/knowledge/query",
            json={"query": query},
        )

        result = data.get("result") or data.get("value") or data.get("answer") or ""
        if not result:
            return []

        # Normalise result to string for page_content
        if isinstance(result, (dict, list)):
            import json
            content = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            content = str(result)

        return [
            Document(
                page_content=content,
                metadata={
                    "source": "cala:knowledge_query",
                    "query": query,
                    "verified": data.get("verified", True),
                    "type": "structured_result",
                    "entity": data.get("entity"),
                    "field": data.get("field"),
                },
            )
        ]

    def _entities(self, query: str) -> List[Document]:
        """GET /v1/entities — entity discovery by name."""
        data = self._make_request(
            "GET",
            "/v1/entities",
            params={"q": query, "limit": self.k},
        )

        entities = data.get("entities") or data.get("results") or []
        docs: List[Document] = []

        for ent in entities[: self.k]:
            # Build a readable summary as page_content
            name = ent.get("name", "")
            entity_type = ent.get("type") or ent.get("entity_type", "")
            description = ent.get("description") or ent.get("summary", "")
            fields: Dict[str, Any] = ent.get("fields") or ent.get("properties") or {}

            lines = []
            if name:
                lines.append(f"**{name}**")
            if entity_type:
                lines.append(f"Type: {entity_type}")
            if description:
                lines.append(description)
            if fields:
                for k, v in list(fields.items())[:8]:  # cap field count
                    lines.append(f"{k}: {v}")

            docs.append(
                Document(
                    page_content="\n".join(lines) or str(ent),
                    metadata={
                        "source": "cala:entities",
                        "entity_id": ent.get("id"),
                        "entity_name": name,
                        "entity_type": entity_type,
                        "verified": ent.get("verified", True),
                        "type": "entity",
                        "query": query,
                    },
                )
            )

        return docs

    # ── LangChain interface ───────────────────────────────────────────────────

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Retrieve documents from Cala for the given query."""
        if self.mode == "search":
            return self._search(query)
        elif self.mode == "query":
            return self._query(query)
        elif self.mode == "entities":
            return self._entities(query)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}")
