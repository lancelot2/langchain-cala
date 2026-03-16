# langchain-cala

**LangChain retriever for [Cala](https://cala.ai) — the verified knowledge graph API for AI agents.**

Cala returns structured, typed, source-cited data instead of raw web scrapes. Every document includes `source` and `verified` metadata fields so your agent can reason over facts with full traceability.

```
pip install langchain-cala
```

---

## Why Cala?

| Web search APIs | Cala |
|---|---|
| Returns URLs + scraped HTML | Returns structured, typed JSON |
| Agent must parse, deduplicate, guess | Agent reasons directly over clean data |
| No source tracing | Every fact cites its source |
| Hallucination risk | Verified entity graph |

---

## Quickstart

```python
from langchain_cala import CalaRetriever

retriever = CalaRetriever(api_key="YOUR_CALA_API_KEY")
docs = retriever.invoke("Latest funding rounds for Barcelona AI startups")

for doc in docs:
    print(doc.page_content)
    print(doc.metadata["source"])
```

Get your API key at [console.cala.ai](https://console.cala.ai).  
You can also set it as an environment variable: `export CALA_API_KEY=your-key`

---

## Modes

`CalaRetriever` supports three modes matching Cala's API:

### `"search"` *(default)* — Natural language questions

Ask a free-text question and get a synthesised answer with source citations.

```python
retriever = CalaRetriever(api_key="...", mode="search")
docs = retriever.invoke("Who are the key investors in European AI infrastructure?")

# docs[0] → synthesised answer
# docs[1:] → source citations with URLs
```

### `"query"` — Structured dot-notation queries

Query the entity graph directly using Cala's dot-notation syntax.

```python
retriever = CalaRetriever(api_key="...", mode="query")
docs = retriever.invoke("OpenAI.founded.year")
# → "2015"

docs = retriever.invoke("Mistral.headquarters.city")
# → "Paris"
```

### `"entities"` — Entity discovery

Find and explore entities by name across companies, people, products, research papers, laws, and places.

```python
retriever = CalaRetriever(api_key="...", mode="entities")
docs = retriever.invoke("Factorial HR")
# → Entity card: name, type, description, fields (funding, location, founded…)
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `CALA_API_KEY` env | Your Cala API key |
| `mode` | `"search" \| "query" \| "entities"` | `"search"` | API mode |
| `k` | `int` | `5` | Max documents returned |
| `timeout` | `int` | `30` | Request timeout (seconds) |

---

## Use in a RAG chain

```python
from langchain_cala import CalaRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

retriever = CalaRetriever(api_key="...")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Every fact in your answer must be traceable to a source.

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(
        f"{doc.page_content}\n[Source: {doc.metadata.get('source', 'cala')}]"
        for doc in docs
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o")
    | StrOutputParser()
)

answer = chain.invoke("Which EU AI startups raised Series A in 2024?")
print(answer)
```

---

## Use as a LangGraph tool

```python
from langchain_cala import CalaRetriever
from langchain_core.tools import create_retriever_tool

retriever = CalaRetriever(api_key="...")

cala_tool = create_retriever_tool(
    retriever,
    name="cala_knowledge",
    description=(
        "Search Cala's verified knowledge graph for structured, "
        "source-cited facts about companies, people, funding, research, and more. "
        "Returns typed data your agent can reason over immediately."
    ),
)

# Add to your LangGraph agent's tools list
tools = [cala_tool, ...]
```

---

## Document metadata

Every `Document` returned includes metadata for full traceability:

```python
{
    "source": "https://techcrunch.com/...",   # or "cala:knowledge_search"
    "verified": True,
    "type": "answer" | "citation" | "structured_result" | "entity",
    "query": "original query string",
    # mode-specific extras:
    "entity": "OpenAI",          # query mode
    "field": "founded.year",     # query mode
    "entity_id": "ent_xxx",      # entities mode
    "entity_type": "Company",    # entities mode
}
```

---

## Installation

```bash
pip install langchain-cala
```

**Requirements:** Python ≥ 3.9, `langchain-core ≥ 0.2`, `requests ≥ 2.28`, `pydantic ≥ 2.0`

---

## Development

```bash
git clone https://github.com/your-username/langchain-cala
cd langchain-cala
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Links

- [Cala documentation](https://docs.cala.ai)
- [Get an API key](https://console.cala.ai)
- [LangChain integrations directory](https://python.langchain.com/docs/integrations/providers/)
- [Report an issue](https://github.com/your-username/langchain-cala/issues)

---

## License

MIT
