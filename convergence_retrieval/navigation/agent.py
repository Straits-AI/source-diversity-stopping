"""
NavigationAgent — the core navigation loop.

This is NOT RAG. RAG = retrieve → generate.
Navigation = explore → decide → explore more → decide → stop → generate.

The agent maintains state (discovery vs knowledge), chooses actions via
a policy, executes them in the environment, and stops when convergence
is detected or budget is exhausted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .state import NavigationState
from .actions import Action, ActionType
from .policy import NavigationPolicy, ConvergencePolicy


@dataclass
class NavigationTrace:
    """Full record of a navigation episode."""
    query: str
    steps: list[dict] = field(default_factory=list)
    final_state: NavigationState | None = None
    stopped_reason: str = ""
    total_ops: int = 0
    elapsed_ms: float = 0.0

    def show(self) -> str:
        """Human-readable trace."""
        lines = [f"Query: {self.query}", ""]
        for step in self.steps:
            action = step["action"]
            lines.append(f"  Step {step['step']}: {action}")
            if step.get("discoveries"):
                for d in step["discoveries"][:3]:
                    lines.append(f"    📄 Discovered: {d['title']} ({d['doc_id']})")
            if step.get("content_read"):
                preview = step["content_read"][:80].replace("\n", " ")
                lines.append(f"    📖 Read: {preview}...")
            if step.get("links_found"):
                lines.append(f"    🔗 Links: {step['links_found'][:3]}")
            state_summary = step.get("state_summary", "")
            if state_summary:
                lines.append(f"    State: {state_summary}")
            lines.append("")

        lines.append(f"Stopped: {self.stopped_reason}")
        lines.append(f"Total ops: {self.total_ops}")
        lines.append(f"Knowledge: {len(self.final_state.known) if self.final_state else 0} items")
        return "\n".join(lines)


@dataclass
class NavigationResult:
    """Result of a navigation episode."""
    knowledge: list[dict]   # what the agent knows (doc_id, content, relevance)
    trace: NavigationTrace
    ops_used: int
    discovered: int         # how many items were found
    read: int              # how many items were actually read


class NavigationAgent:
    """
    Agent that navigates a heterogeneous information environment.

    Unlike a retriever (which fetches and returns), the agent:
    1. Explores the environment step by step
    2. Maintains state: what it has DISCOVERED vs what it KNOWS
    3. Decides at each step: go deeper, go broader, or stop
    4. Uses convergence as one stopping signal

    Parameters
    ----------
    environment : Environment or DocumentEnvironment
        The information environment to navigate.
    policy : NavigationPolicy, optional
        The navigation policy. Default: ConvergencePolicy.
    max_steps : int
        Maximum navigation steps.
    budget : float
        Total budget (each action costs some fraction of this).

    Example
    -------
    >>> from convergence_retrieval import BM25Substrate, DenseSubstrate
    >>> from convergence_retrieval.environments import DocumentEnvironment
    >>> from convergence_retrieval.navigation import NavigationAgent
    >>>
    >>> env = DocumentEnvironment(substrates=[BM25Substrate(), DenseSubstrate()])
    >>> env.load(documents)
    >>> agent = NavigationAgent(environment=env)
    >>> result = agent.navigate("How does auth middleware validate tokens?")
    >>> print(result.trace.show())
    """

    def __init__(
        self,
        environment,  # any object with .execute(action), .reset(), .substrate_names
        policy: NavigationPolicy | None = None,
        max_steps: int = 10,
        budget: float = 10.0,
    ) -> None:
        self._env = environment
        self._max_steps = max_steps
        self._budget = budget

        # Default policy: convergence with environment's substrate order
        if policy is None:
            substrate_names = environment.substrate_names
            self._policy = ConvergencePolicy(
                substrate_order=substrate_names,
                min_sources=min(2, len(substrate_names)),
                max_steps=max_steps,
            )
        else:
            self._policy = policy

    def navigate(self, query: str) -> NavigationResult:
        """
        Navigate the environment to gather evidence for a query.

        This is the core loop:
          1. Policy chooses an action based on current state
          2. Environment executes the action
          3. State is updated with discoveries and knowledge
          4. Repeat until policy says STOP or budget exhausted

        Returns
        -------
        NavigationResult
            Contains the gathered knowledge, a full navigation trace,
            and statistics.
        """
        t0 = time.perf_counter()

        # Initialize state
        state = NavigationState(query=query, budget_remaining=self._budget)

        # Reset policy for new episode
        self._policy.reset()

        trace = NavigationTrace(query=query)

        for step in range(self._max_steps):
            state.step = step

            # Policy chooses action
            action = self._policy.choose_action(state)

            # Record step
            step_record = {
                "step": step,
                "action": str(action),
                "action_type": action.type.value,
            }

            # STOP
            if action.type == ActionType.STOP:
                trace.stopped_reason = self._stop_reason(state)
                step_record["state_summary"] = state.summary()
                trace.steps.append(step_record)
                break

            # Execute action
            result = self._env.execute(action)
            state.budget_remaining -= result.cost
            trace.total_ops += 1

            # Update state with discoveries
            for d in result.discoveries:
                new = state.add_discovery(
                    doc_id=d["doc_id"],
                    title=d.get("title", d["doc_id"]),
                    snippet=d.get("snippet", ""),
                    source=d.get("source", action.type.value),
                )

            step_record["discoveries"] = [
                {"doc_id": d["doc_id"], "title": d.get("title", "")}
                for d in result.discoveries
            ]

            # Update state with knowledge (if content was read)
            if result.content_read:
                doc_id = action.params.get("doc_id", action.params.get("link", ""))
                if doc_id:
                    # Compute relevance as simple keyword overlap
                    relevance = self._compute_relevance(query, result.content_read)
                    source = action.params.get("substrate", action.type.value)
                    state.add_knowledge(
                        doc_id=doc_id,
                        content=result.content_read,
                        extracted=result.content_read[:200],
                        source=source,
                        relevance=relevance,
                    )

                step_record["content_read"] = result.content_read[:100]

            if result.links_found:
                step_record["links_found"] = result.links_found[:5]
                # Also record links in history for policy to see
                state.history.append({
                    "step": step,
                    "action": str(action),
                    "links_found": result.links_found,
                    "n_discoveries": len(result.discoveries),
                })
            else:
                state.history.append({
                    "step": step,
                    "action": str(action),
                    "n_discoveries": len(result.discoveries),
                })

            step_record["state_summary"] = f"D={state.n_discovered} K={state.n_known} sources={state.knowledge_sources}"
            trace.steps.append(step_record)

            # Budget check
            if state.budget_remaining <= 0:
                trace.stopped_reason = "budget_exhausted"
                break
        else:
            trace.stopped_reason = "max_steps"

        trace.elapsed_ms = (time.perf_counter() - t0) * 1000
        trace.final_state = state

        # Gather knowledge for output
        knowledge = [
            {
                "doc_id": k.doc_id,
                "content": k.content,
                "relevance": k.relevance,
                "source": k.source,
            }
            for k in sorted(
                state.known.values(),
                key=lambda k: k.relevance,
                reverse=True,
            )
        ]

        return NavigationResult(
            knowledge=knowledge,
            trace=trace,
            ops_used=trace.total_ops,
            discovered=state.n_discovered,
            read=state.n_known,
        )

    def _stop_reason(self, state: NavigationState) -> str:
        if len(state.knowledge_sources) >= 2:
            return "convergence"
        if state.budget_remaining <= 0:
            return "budget"
        return "policy_decision"

    @staticmethod
    def _compute_relevance(query: str, content: str) -> float:
        """Simple keyword overlap relevance."""
        q_tokens = set(query.lower().split())
        c_tokens = set(content.lower().split())
        if not q_tokens:
            return 0.0
        overlap = q_tokens & c_tokens
        return len(overlap) / len(q_tokens)
