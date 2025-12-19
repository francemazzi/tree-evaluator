"""
Paper search tool - fetches scientific papers from arXiv and PubMed.
Returns abstracts with source links for RAG pipelines.
"""

from __future__ import annotations

from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class PaperSearchInput(BaseModel):
    """Input schema for paper search tool."""

    query: str = Field(description="Search query (natural language or keywords)")
    source: str = Field(
        default="arxiv",
        description="Source: 'arxiv', 'pubmed', or 'both'",
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of papers to return (1-10)",
    )


class PaperSearchTool(BaseTool):
    """Search arXiv and PubMed for scientific papers. Returns abstracts with links."""

    name: str = "search_scientific_papers"
    description: str = """
Search arXiv and/or PubMed for scientific papers.

Returns for each paper:
- title
- authors
- abstract (summary)
- link (URL to full paper)
- source (arxiv or pubmed)

Use this when:
- User asks about scientific research, publications, or literature
- User wants to find papers on a specific topic
- User needs citations or references for a claim

Parameters:
- query: what to search for (natural language or keywords)
- source: 'arxiv' (preprints, physics/CS/math), 'pubmed' (biomedical), or 'both'
- max_results: how many papers (default 5, max 10)
"""
    args_schema: Type[BaseModel] = PaperSearchInput

    def _search_arxiv(self, query: str, max_results: int) -> List[dict]:
        """Fetch papers from arXiv."""
        try:
            from langchain_community.retrievers import ArxivRetriever
        except ImportError:
            return [{"error": "arxiv package not installed. Run: pip install arxiv pymupdf"}]

        try:
            retriever = ArxivRetriever(
                load_max_docs=max_results,
                get_full_documents=False,  # abstracts only = faster
            )
            docs = retriever.invoke(query)

            results = []
            for doc in docs:
                meta = doc.metadata
                entry_id = meta.get("entry_id", meta.get("Entry ID", ""))
                # arXiv entry_id is like "http://arxiv.org/abs/2301.12345v1"
                link = entry_id if entry_id.startswith("http") else f"https://arxiv.org/abs/{entry_id}"

                results.append({
                    "title": meta.get("Title", meta.get("title", "N/A")),
                    "authors": meta.get("Authors", meta.get("authors", "N/A")),
                    "abstract": doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content,
                    "link": link,
                    "source": "arxiv",
                    "published": meta.get("Published", meta.get("published", "")),
                })
            return results

        except Exception as e:
            return [{"error": f"arXiv search failed: {str(e)}"}]

    def _search_pubmed(self, query: str, max_results: int) -> List[dict]:
        """Fetch papers from PubMed."""
        try:
            from langchain_community.retrievers import PubMedRetriever
        except ImportError:
            return [{"error": "PubMed package not installed. Run: pip install xmltodict"}]

        try:
            retriever = PubMedRetriever(top_k_results=max_results)
            docs = retriever.invoke(query)

            results = []
            for doc in docs:
                meta = doc.metadata
                uid = meta.get("uid", "")
                link = f"https://pubmed.ncbi.nlm.nih.gov/{uid}/" if uid else ""

                results.append({
                    "title": meta.get("Title", meta.get("title", "N/A")),
                    "authors": meta.get("Authors", meta.get("authors", "N/A")),
                    "abstract": doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content,
                    "link": link,
                    "source": "pubmed",
                    "published": meta.get("Published", meta.get("published", "")),
                })
            return results

        except Exception as e:
            return [{"error": f"PubMed search failed: {str(e)}"}]

    def _run(
        self,
        query: str,
        source: str = "arxiv",
        max_results: int = 5,
    ) -> dict:
        """Execute paper search."""
        max_results = min(max(1, max_results), 10)
        source = source.lower().strip()

        papers: List[dict] = []

        if source in ("arxiv", "both"):
            papers.extend(self._search_arxiv(query, max_results))

        if source in ("pubmed", "both"):
            papers.extend(self._search_pubmed(query, max_results))

        if not papers:
            return {
                "query": query,
                "source": source,
                "papers": [],
                "message": "No papers found for this query.",
            }

        # Check for errors
        errors = [p for p in papers if "error" in p]
        valid = [p for p in papers if "error" not in p]

        return {
            "query": query,
            "source": source,
            "paper_count": len(valid),
            "papers": valid,
            "errors": errors if errors else None,
        }


