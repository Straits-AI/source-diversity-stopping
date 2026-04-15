"""
Navigation state — the agent's evolving understanding of the information environment.

Implements the discovery/knowledge split from information foraging theory:
- Discovery: what the agent knows EXISTS (titles, paths, mentions) but hasn't read
- Knowledge: what the agent has actually READ and verified
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiscoveryEntry:
    """Something the agent knows exists but hasn't fully examined."""
    doc_id: str
    title: str
    snippet: str  # first ~100 chars or summary
    source: str   # which substrate/action found it
    step: int     # when it was discovered

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        return isinstance(other, DiscoveryEntry) and self.doc_id == other.doc_id


@dataclass
class KnowledgeEntry:
    """Something the agent has actually read and extracted."""
    doc_id: str
    content: str      # the full text read
    extracted: str     # key information extracted (summary/claims)
    source: str        # how it was accessed
    step: int          # when it was read
    relevance: float   # how relevant it is to the query (0-1)

    def __hash__(self):
        return hash(self.doc_id)

    def __eq__(self, other):
        return isinstance(other, KnowledgeEntry) and self.doc_id == other.doc_id


@dataclass
class NavigationState:
    """
    The agent's full state at any point during navigation.

    s_t = (query, discovered, known, history, budget_remaining)

    This IS the agentic attention state from the original vision —
    now implemented as a practical, trackable object.
    """
    query: str
    discovered: dict[str, DiscoveryEntry] = field(default_factory=dict)
    known: dict[str, KnowledgeEntry] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    step: int = 0
    budget_remaining: float = 1.0

    @property
    def n_discovered(self) -> int:
        """How many items the agent knows exist."""
        return len(self.discovered)

    @property
    def n_known(self) -> int:
        """How many items the agent has actually read."""
        return len(self.known)

    @property
    def discovery_sources(self) -> set[str]:
        """Which substrates/actions have contributed discoveries."""
        return {e.source for e in self.discovered.values()}

    @property
    def knowledge_sources(self) -> set[str]:
        """Which substrates/actions have contributed knowledge."""
        return {e.source for e in self.known.values()}

    @property
    def unexplored(self) -> list[DiscoveryEntry]:
        """Items discovered but not yet read."""
        return [d for d in self.discovered.values() if d.doc_id not in self.known]

    def add_discovery(self, doc_id: str, title: str, snippet: str, source: str) -> bool:
        """
        Record that a document was discovered. Returns True if new.

        Discovery = "I know this file/document/section exists"
        This is information SCENT (Pirolli & Card, 1999).
        """
        if doc_id in self.discovered:
            return False
        self.discovered[doc_id] = DiscoveryEntry(
            doc_id=doc_id, title=title, snippet=snippet,
            source=source, step=self.step,
        )
        return True

    def add_knowledge(self, doc_id: str, content: str, extracted: str,
                      source: str, relevance: float) -> bool:
        """
        Record that a document was read and information extracted. Returns True if new.

        Knowledge = "I have read this and know what it says"
        This is information DIET (Pirolli & Card, 1999).
        """
        if doc_id in self.known:
            return False
        self.known[doc_id] = KnowledgeEntry(
            doc_id=doc_id, content=content, extracted=extracted,
            source=source, step=self.step, relevance=relevance,
        )
        return True

    def summary(self) -> str:
        """Human-readable state summary."""
        lines = [
            f"Step {self.step} | Budget: {self.budget_remaining:.0%}",
            f"Discovered: {self.n_discovered} items from {self.discovery_sources}",
            f"Known: {self.n_known} items from {self.knowledge_sources}",
            f"Unexplored: {len(self.unexplored)} items",
        ]
        return "\n".join(lines)
