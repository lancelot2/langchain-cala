"""langchain-cala: LangChain integration for the Cala verified knowledge graph API.

Quickstart::

    from langchain_cala import CalaRetriever

    retriever = CalaRetriever(api_key="YOUR_CALA_API_KEY")
    docs = retriever.invoke("Latest funding rounds for Barcelona AI startups")
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
"""

from langchain_cala.retriever import CalaRetriever

__version__ = "0.1.0"
__all__ = ["CalaRetriever"]
