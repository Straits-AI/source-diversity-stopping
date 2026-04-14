"""
Executable address space.

Executes simple computations on structured data found in passages.
Numbers, quantities, and measurements are extracted via regex from the corpus;
the TOOL_CALL operation then answers comparison and arithmetic questions
directly from the workspace.

Supported operations: SEARCH, TOOL_CALL
"""

from __future__ import annotations

import re
import time
from typing import Optional

from ..types import AgentState, ActionResult, Operation
from .base import AddressSpace

# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns for numeric quantity extraction
# ─────────────────────────────────────────────────────────────────────────────

# Pattern priority (high → low):
#   1. Currency: $X [billion/million/…]
#   2. Percentage: X%
#   3. "population of X [thousand/million]"
#   4. "X [billion/million/thousand/…]" (plain magnitude)
#   5. Comma-grouped large integers (≥ 4 digits or with comma grouping, excl. years)

_P_CURRENCY = re.compile(
    r"\$\s*(?P<val>[\d,]+(?:\.\d+)?)\s*(?P<unit>trillion|billion|million|thousand|k\b)?",
    re.IGNORECASE,
)
_P_PERCENT = re.compile(
    r"(?P<val>[\d,]+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_P_POPULATION = re.compile(
    r"\bpopulation\s+(?:of|:)\s*(?P<val>[\d,]+(?:\.\d+)?)\s*(?P<unit>trillion|billion|million|thousand)?",
    re.IGNORECASE,
)
_P_INHABITANTS = re.compile(
    # "home to X,000 inhabitants" / "X,000 residents"
    r"\bhome\s+to\s+(?P<val>[\d,]+(?:\.\d+)?)\s+(?:inhabitants|residents|people)\b"
    r"|(?P<val2>[\d,]+(?:\.\d+)?)\s+(?:inhabitants|residents|people)\b",
    re.IGNORECASE,
)
_P_MAGNITUDE = re.compile(
    r"(?<!\$)\b(?P<val>[\d,]+(?:\.\d+)?)\s+(?P<unit>trillion|billion|million|thousand)\b",
    re.IGNORECASE,
)
# Comma-grouped numbers like 42,000 / 1,200,000 (must have at least one comma group)
_P_COMMA_NUMBER = re.compile(
    r"\b(?P<val>\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b",
)

# Year range filter — 4-digit numbers in [1700, 2100] are likely years
_YEAR_RE = re.compile(r"^\d{4}$")


def _is_year(val: float) -> bool:
    """Return True if *val* looks like a calendar year."""
    return 1700.0 <= val <= 2100.0 and val == int(val) and len(str(int(val))) == 4


# Multipliers for unit suffixes
_UNIT_MULTIPLIERS: dict[str, float] = {
    "trillion": 1e12,
    "billion":  1e9,
    "million":  1e6,
    "thousand": 1e3,
    "k":        1e3,
}

DEFAULT_TOP_K = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_number(value_str: str, unit: Optional[str] = None) -> float:
    """Convert matched value + optional unit suffix to a float."""
    cleaned = value_str.replace(",", "")
    base = float(cleaned)
    if unit:
        mult = _UNIT_MULTIPLIERS.get(unit.lower(), 1.0)
        base *= mult
    return base


def _extract_quantities(text: str) -> list[dict]:
    """
    Return a list of quantity records found in *text*.

    Each record:
        {"raw": str, "value": float, "context": str, "is_year": bool}

    Records for year-like values are included but flagged; callers can filter
    them for arithmetic but may still use them to detect numeric passages.
    """
    found: list[dict] = []
    seen_spans: set[tuple[int, int]] = set()

    def _add(m: re.Match, val_group: str = "val", unit_group: Optional[str] = "unit") -> None:
        span = m.span()
        if any(s <= span[0] < e for s, e in seen_spans):
            return
        try:
            unit = m.groupdict().get(unit_group) if unit_group else None
            v = _parse_number(m.group(val_group), unit)
        except (ValueError, IndexError, AttributeError):
            return
        seen_spans.add(span)
        start = max(0, span[0] - 40)
        end = min(len(text), span[1] + 40)
        ctx = text[start:end].strip()
        found.append({
            "raw": m.group(0),
            "value": v,
            "context": ctx,
            "is_year": _is_year(v),
        })

    for m in _P_CURRENCY.finditer(text):
        _add(m)
    for m in _P_PERCENT.finditer(text):
        _add(m, "val", None)
    for m in _P_POPULATION.finditer(text):
        _add(m)
    # inhabitants pattern has two named groups (val / val2)
    for m in _P_INHABITANTS.finditer(text):
        span = m.span()
        if any(s <= span[0] < e for s, e in seen_spans):
            continue
        raw_val = m.group("val") or m.group("val2")
        if raw_val is None:
            continue
        try:
            v = _parse_number(raw_val, None)
        except ValueError:
            continue
        seen_spans.add(span)
        start = max(0, span[0] - 40)
        end = min(len(text), span[1] + 40)
        ctx = text[start:end].strip()
        found.append({"raw": m.group(0), "value": v, "context": ctx, "is_year": _is_year(v)})
    for m in _P_MAGNITUDE.finditer(text):
        _add(m)
    for m in _P_COMMA_NUMBER.finditer(text):
        _add(m, "val", None)

    return found


def _has_quantities(text: str) -> bool:
    """Return True if *text* contains at least one recognisable quantity."""
    return bool(_extract_quantities(text))


def _score_document(query_text: str, doc_content: str) -> float:
    """
    Lightweight relevance score for SEARCH: fraction of query keywords present
    in the document, boosted by presence of any non-year quantities.
    """
    q_lower = query_text.lower()
    d_lower = doc_content.lower()

    _STOP = frozenset(
        {"a","an","the","is","are","was","were","be","been","have","has",
         "had","do","does","did","will","would","could","should","may",
         "might","can","in","on","at","by","for","with","from","to","of",
         "and","or","but","so","yet","nor","not","this","that","it","its",
         "what","which","who","whom","where","when","why","how"}
    )
    q_tokens = [t for t in re.findall(r"\b\w+\b", q_lower) if t not in _STOP and len(t) > 2]
    if not q_tokens:
        return 0.1 if _has_quantities(doc_content) else 0.0

    hits = sum(1 for t in q_tokens if t in d_lower)
    keyword_score = hits / len(q_tokens)

    # Bonus only for non-year quantities
    non_year_qtys = [q for q in _extract_quantities(doc_content) if not q["is_year"]]
    qty_bonus = 0.2 if non_year_qtys else 0.0
    return min(1.0, keyword_score + qty_bonus)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL_CALL computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_non_year_qty(content: str) -> Optional[dict]:
    """Return the highest-value non-year quantity in *content*, or None."""
    qtys = [q for q in _extract_quantities(content) if not q["is_year"] and q["value"] > 0]
    if not qtys:
        return None
    return max(qtys, key=lambda x: x["value"])


def _tool_comparison(workspace_items: list[dict], query_text: str) -> dict:
    """
    Answer "which X is larger/higher?" questions.

    Strategy:
    - For each workspace item collect the largest non-year quantity.
    - Identify the two items with the highest quantities.
    - The entity name is taken from the item's ``title`` field when present,
      falling back to the ``id`` field.
    - Return a structured answer.
    """
    candidates: list[dict] = []
    for item in workspace_items:
        content = item.get("content", "")
        qty = _best_non_year_qty(content)
        if qty is None:
            continue
        # Prefer title over raw id for human-readable answers
        label = item.get("title") or item.get("id", item.get("source_id", ""))
        candidates.append({
            "label": label,
            "value": qty["value"],
            "raw": qty["raw"],
        })

    if len(candidates) < 2:
        return {
            "answer": "insufficient data",
            "computation": "fewer than 2 numeric passages found in workspace",
            "success": False,
        }

    candidates.sort(key=lambda x: x["value"], reverse=True)
    larger = candidates[0]
    smaller = candidates[1]
    answer = larger["label"]
    computation = (
        f"{larger['label']}: {larger['value']:,.0f} "
        f"vs {smaller['label']}: {smaller['value']:,.0f}. "
        f"Larger: {larger['label']}"
    )
    return {"answer": answer, "computation": computation, "success": True}


def _tool_arithmetic(workspace_items: list[dict], query_text: str) -> dict:
    """
    Answer "what is the combined/total?" questions.

    For each distinct workspace item, take the largest non-year quantity.
    Sum them and report.
    """
    by_source: dict[str, float] = {}
    by_source_raw: dict[str, str] = {}
    by_source_label: dict[str, str] = {}

    for item in workspace_items:
        doc_id = item.get("id", item.get("source_id", ""))
        content = item.get("content", "")
        qty = _best_non_year_qty(content)
        if qty is None:
            continue
        # Prefer keeping the highest value we found for this source
        if doc_id not in by_source or qty["value"] > by_source[doc_id]:
            by_source[doc_id] = qty["value"]
            by_source_raw[doc_id] = qty["raw"]
            label = item.get("title") or doc_id
            by_source_label[doc_id] = label

    if not by_source:
        return {
            "answer": "insufficient data",
            "computation": "no numeric quantities found in workspace",
            "success": False,
        }

    total = sum(by_source.values())

    # Report total in thousands if all values are in thousands range [1,000 – 999,999]
    all_in_thousands = all(1_000 <= v <= 999_999 for v in by_source.values())
    if all_in_thousands:
        total_display = int(round(total / 1_000))
        parts = " + ".join(
            f"{by_source_label[src]}({by_source_raw[src]}={int(v/1000)}k)"
            for src, v in by_source.items()
        )
        answer = str(total_display)
        computation = f"{parts} = {total_display} (thousands)"
    else:
        parts = " + ".join(
            f"{by_source_label[src]}({by_source_raw[src]}={v:,.0f})"
            for src, v in by_source.items()
        )
        answer = f"{total:,.0f}"
        computation = f"{parts} = {total:,.0f}"

    return {"answer": answer, "computation": computation, "success": True}


def _tool_percentage(workspace_items: list[dict], query_text: str) -> dict:
    """
    Answer "what percentage is X of Y?" questions.
    Takes the two distinct-source quantities and computes (smaller/larger)*100.
    """
    by_source: dict[str, float] = {}
    by_source_raw: dict[str, str] = {}

    for item in workspace_items:
        doc_id = item.get("id", item.get("source_id", ""))
        content = item.get("content", "")
        qty = _best_non_year_qty(content)
        if qty is None:
            continue
        if doc_id not in by_source:
            by_source[doc_id] = qty["value"]
            by_source_raw[doc_id] = qty["raw"]

    values = sorted(by_source.values(), reverse=True)
    if len(values) < 2:
        return {
            "answer": "insufficient data",
            "computation": "fewer than 2 numeric sources found in workspace",
            "success": False,
        }

    numerator, denominator = values[1], values[0]
    if denominator == 0:
        return {
            "answer": "division by zero",
            "computation": "denominator is 0",
            "success": False,
        }

    pct = (numerator / denominator) * 100.0
    answer = f"{pct:.2f}%"
    computation = f"({numerator:,.0f} / {denominator:,.0f}) * 100 = {pct:.2f}%"
    return {"answer": answer, "computation": computation, "success": True}


_COMPARISON_RE = re.compile(
    r"\b(larger|higher|greater|more|bigger|most|largest|highest|greatest|best|"
    r"smaller|lower|less|least|smallest|lowest|worst|compare|comparison|between)\b",
    re.IGNORECASE,
)
_ARITHMETIC_RE = re.compile(
    r"\b(total|combined|sum|altogether|together|add|plus|aggregate|overall)\b",
    re.IGNORECASE,
)
_PERCENTAGE_RE = re.compile(r"\bpercentage\b|\bpercent\b|\bwhat\s+percent", re.IGNORECASE)


def _detect_computation_type(query_text: str) -> str:
    """Classify the computation the query requires."""
    if _PERCENTAGE_RE.search(query_text):
        return "percentage"
    if _ARITHMETIC_RE.search(query_text):
        return "arithmetic"
    if _COMPARISON_RE.search(query_text):
        return "comparison"
    return "comparison"  # default


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class ExecutableAddressSpace(AddressSpace):
    """
    Executable address space — SEARCH + TOOL_CALL.

    SEARCH
    ------
    Finds passages that contain numbers/quantities and are relevant to the
    query.  Passages are scored by keyword overlap with the query, with a
    bonus for containing recognised non-year numeric patterns.

    TOOL_CALL
    ---------
    Operates on workspace items already loaded by previous SEARCH steps.
    Detects the computation type from the query string and runs:

    * comparison  — "which X is larger/higher?"
    * arithmetic  — "what is the combined/total?"
    * percentage  — "what percentage is X of Y?"

    The result item contains both the ``answer`` and the full ``computation``
    trace so downstream evaluation can verify correctness.

    Parameters
    ----------
    top_k : int
        Default number of search results.  Default 5.
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K) -> None:
        self._top_k = top_k
        self._documents: list[dict] = []
        self._is_built: bool = False

    # ─────────────────────────────────────────────────────────
    # AddressSpace interface
    # ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "executable"

    @property
    def supported_operations(self) -> list[Operation]:
        return [Operation.SEARCH, Operation.TOOL_CALL]

    def build_index(self, documents: list[dict]) -> None:
        """
        Store documents.  No heavy index is required; scoring is on-the-fly.

        Parameters
        ----------
        documents : list[dict]
            Must contain ``"id"`` and ``"content"``.
        """
        self._documents = list(documents)
        self._is_built = True

    def query(
        self,
        state: AgentState,
        operation: Operation,
        params: dict,
    ) -> ActionResult:
        self._assert_operation(operation)

        if not self._is_built:
            return ActionResult(
                items=[], cost_tokens=0, cost_latency_ms=0.0, success=False,
                error="Index has not been built. Call build_index() first.",
            )

        t0 = time.perf_counter()

        if operation == Operation.SEARCH:
            result = self._do_search(state, params, t0)
        else:  # TOOL_CALL
            result = self._do_tool_call(state, params, t0)

        return result

    # ─────────────────────────────────────────────────────────
    # SEARCH implementation
    # ─────────────────────────────────────────────────────────

    def _do_search(self, state: AgentState, params: dict, t0: float) -> ActionResult:
        query_text: str = params.get("query", state.query)
        top_k: int = int(params.get("top_k", self._top_k))

        if not self._documents:
            return ActionResult(
                items=[],
                cost_tokens=self._estimate_tokens(query_text),
                cost_latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # Score every document; prefer those with non-year quantities
        scored: list[tuple[float, dict]] = []
        no_qty: list[tuple[float, dict]] = []

        for doc in self._documents:
            content = doc.get("content", "")
            score = _score_document(query_text, content)
            non_year = [q for q in _extract_quantities(content) if not q["is_year"]]
            if non_year:
                scored.append((score, doc))
            else:
                no_qty.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results_list = scored[:top_k]

        # If not enough, fill with non-quantity docs
        if len(results_list) < top_k:
            no_qty.sort(key=lambda x: x[0], reverse=True)
            results_list += no_qty[: top_k - len(results_list)]

        items = []
        for score, doc in results_list:
            item = {k: v for k, v in doc.items()}
            item["score"] = score
            item["source_type"] = "executable_passage"
            items.append(item)

        cost_tokens = (
            self._estimate_tokens(query_text)
            + self._estimate_tokens(" ".join(it["content"] for it in items))
        )
        return ActionResult(
            items=items,
            cost_tokens=cost_tokens,
            cost_latency_ms=(time.perf_counter() - t0) * 1000,
            cost_operations=1,
        )

    # ─────────────────────────────────────────────────────────
    # TOOL_CALL implementation
    # ─────────────────────────────────────────────────────────

    def _do_tool_call(self, state: AgentState, params: dict, t0: float) -> ActionResult:
        """
        Execute a computation over workspace items.

        Params accepted (all optional):
          ``query``         — override for computation-type detection
          ``computation``   — explicit type: "comparison" | "arithmetic" | "percentage"
          ``workspace``     — explicit list of item dicts (defaults to state.workspace)
        """
        query_text: str = params.get("query", state.query)

        # Use explicit workspace items if provided, else fall back to state
        raw_items = params.get("workspace", None)
        if raw_items is None:
            workspace_items = [
                {
                    "id": w.source_id,
                    "content": w.content,
                    "score": w.relevance_score,
                    # surface title if embedded in source_id by harness
                }
                for w in state.workspace
            ]
            # Enrich with title from index if available
            _id_to_doc = {doc["id"]: doc for doc in self._documents}
            for wi in workspace_items:
                doc = _id_to_doc.get(wi["id"], {})
                if "title" in doc:
                    wi["title"] = doc["title"]
        else:
            workspace_items = raw_items

        comp_type = params.get("computation", _detect_computation_type(query_text))

        if comp_type == "percentage":
            result_data = _tool_percentage(workspace_items, query_text)
        elif comp_type == "arithmetic":
            result_data = _tool_arithmetic(workspace_items, query_text)
        else:
            result_data = _tool_comparison(workspace_items, query_text)

        # Pack result as a single item so the harness can add it to workspace
        answer_text = (
            f"[COMPUTATION: {comp_type}] "
            f"Answer: {result_data['answer']}. "
            f"Computation: {result_data['computation']}"
        )
        item = {
            "id": f"tool_result_{state.step}",
            "content": answer_text,
            "score": 1.0 if result_data.get("success") else 0.0,
            "source_type": "tool_result",
            "computation_type": comp_type,
            "answer": result_data["answer"],
            "computation": result_data["computation"],
            "success": result_data.get("success", False),
        }

        cost_tokens = self._estimate_tokens(query_text) + self._estimate_tokens(answer_text)
        return ActionResult(
            items=[item],
            cost_tokens=cost_tokens,
            cost_latency_ms=(time.perf_counter() - t0) * 1000,
            cost_operations=1,
        )
