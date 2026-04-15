"""
Navigation actions — what the agent can DO in the information environment.

Unlike RAG (which has one action: retrieve), navigation has:
- SEARCH: find documents matching a query (via any substrate)
- OPEN: read a discovered document fully
- READ_SECTION: read a specific section/function within a document
- FOLLOW_LINK: follow a reference/import/citation to another document
- STOP: conclude navigation, answer from current knowledge
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    SEARCH = "search"           # search via a substrate
    OPEN = "open"               # read a discovered document
    READ_SECTION = "read_section"  # read a specific section within a document
    FOLLOW_LINK = "follow_link"    # follow a reference to another document
    STOP = "stop"               # stop navigating


@dataclass
class Action:
    """An action the agent wants to take."""
    type: ActionType
    params: dict = field(default_factory=dict)
    # params examples:
    #   SEARCH: {"query": "auth middleware", "substrate": "bm25"}
    #   OPEN: {"doc_id": "auth.py"}
    #   READ_SECTION: {"doc_id": "auth.py", "section": "validate_token"}
    #   FOLLOW_LINK: {"from_doc": "auth.py", "link": "import jwt_utils"}
    #   STOP: {}

    def __repr__(self):
        if self.type == ActionType.STOP:
            return "STOP"
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{self.type.value}({params_str})"


@dataclass
class ActionResult:
    """What the environment returns after an action."""
    success: bool
    discoveries: list[dict] = field(default_factory=list)
    # each discovery: {"doc_id": ..., "title": ..., "snippet": ..., "score": ...}
    content_read: str = ""      # full content if OPEN/READ_SECTION
    links_found: list[str] = field(default_factory=list)  # references found in content
    cost: float = 1.0           # cost of this action in budget units
    error: str = ""
