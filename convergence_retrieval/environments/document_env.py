"""
Document environment — a collection of documents with multiple search substrates.

Supports: SEARCH (via substrates), OPEN (read full doc), FOLLOW_LINK (cross-references).
"""

from __future__ import annotations

import re

from ..substrates.base import Substrate

# Import action types locally to avoid circular imports
from ..navigation.actions import ActionType, ActionResult


class DocumentEnvironment:
    """
    An environment of documents searchable via multiple substrates.

    This wraps multiple retrieval backends and exposes them through
    the navigation action interface (search, open, read_section, follow_link).
    """

    def __init__(self, substrates: list[Substrate]) -> None:
        self._substrates = {s.name: s for s in substrates}
        self._docs: dict[str, dict] = {}  # doc_id → document
        self._links: dict[str, list[str]] = {}  # doc_id → linked doc_ids

    @property
    def substrate_names(self) -> list[str]:
        return list(self._substrates.keys())

    def load(self, documents: list[dict]) -> None:
        """
        Load documents and build all indices.

        Parameters
        ----------
        documents : list[dict]
            Each dict: {"id": str, "content": str, "title": str (optional),
                        "links": list[str] (optional — doc_ids this doc references)}
        """
        self._docs = {doc["id"]: doc for doc in documents}

        # Build link graph
        for doc in documents:
            self._links[doc["id"]] = doc.get("links", [])
            # Also auto-detect links: any mention of another doc's id in the content
            for other in documents:
                if other["id"] != doc["id"]:
                    if other["id"] in doc["content"] or other.get("title", "") in doc["content"]:
                        if other["id"] not in self._links[doc["id"]]:
                            self._links[doc["id"]].append(other["id"])

        # Index all substrates
        for substrate in self._substrates.values():
            substrate.index(documents)

    def reset(self) -> None:
        pass  # stateless — no per-episode state in the environment

    def execute(self, action: Action) -> ActionResult:
        """Execute a navigation action."""
        if action.type == ActionType.SEARCH:
            return self._do_search(action)
        elif action.type == ActionType.OPEN:
            return self._do_open(action)
        elif action.type == ActionType.READ_SECTION:
            return self._do_read_section(action)
        elif action.type == ActionType.FOLLOW_LINK:
            return self._do_follow_link(action)
        elif action.type == ActionType.STOP:
            return ActionResult(success=True, cost=0)
        else:
            return ActionResult(success=False, error=f"Unknown action: {action.type}")

    def _do_search(self, action: Action) -> ActionResult:
        query = action.params.get("query", "")
        substrate_name = action.params.get("substrate")

        if substrate_name and substrate_name in self._substrates:
            substrates = [self._substrates[substrate_name]]
        else:
            # Search the first available substrate
            substrates = [list(self._substrates.values())[0]]

        discoveries = []
        for substrate in substrates:
            results = substrate.search(query, top_k=5)
            for r in results:
                doc = self._docs.get(r.doc_id, {})
                discoveries.append({
                    "doc_id": r.doc_id,
                    "title": doc.get("title", r.doc_id),
                    "snippet": r.content[:150],
                    "score": r.score,
                    "source": substrate.name,
                })

        return ActionResult(
            success=True,
            discoveries=discoveries,
            cost=1.0,
        )

    def _do_open(self, action: Action) -> ActionResult:
        doc_id = action.params.get("doc_id", "")
        doc = self._docs.get(doc_id)

        if not doc:
            return ActionResult(success=False, error=f"Document not found: {doc_id}", cost=0.5)

        links = self._links.get(doc_id, [])

        return ActionResult(
            success=True,
            content_read=doc["content"],
            links_found=links,
            discoveries=[{
                "doc_id": doc_id,
                "title": doc.get("title", doc_id),
                "snippet": doc["content"][:150],
                "score": 1.0,
                "source": "open",
            }],
            cost=1.0,
        )

    def _do_read_section(self, action: Action) -> ActionResult:
        doc_id = action.params.get("doc_id", "")
        section = action.params.get("section", "")
        doc = self._docs.get(doc_id)

        if not doc:
            return ActionResult(success=False, error=f"Document not found: {doc_id}", cost=0.5)

        # Find the section in the content (simple: search for the section name)
        content = doc["content"]
        section_lower = section.lower()

        # Try to find a paragraph containing the section keyword
        paragraphs = content.split("\n\n")
        matching = [p for p in paragraphs if section_lower in p.lower()]
        section_content = "\n\n".join(matching) if matching else content[:500]

        return ActionResult(
            success=True,
            content_read=section_content,
            cost=0.5,  # cheaper than reading the whole doc
        )

    def _do_follow_link(self, action: Action) -> ActionResult:
        link = action.params.get("link", "")

        # Try to find the linked document
        if link in self._docs:
            doc = self._docs[link]
            links = self._links.get(link, [])
            return ActionResult(
                success=True,
                content_read=doc["content"],
                links_found=links,
                discoveries=[{
                    "doc_id": link,
                    "title": doc.get("title", link),
                    "snippet": doc["content"][:150],
                    "score": 1.0,
                    "source": "follow_link",
                }],
                cost=1.0,
            )

        return ActionResult(success=False, error=f"Link target not found: {link}", cost=0.5)
