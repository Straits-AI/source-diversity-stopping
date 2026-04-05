# Adaptive Retrieval Routing: When Knowing What Not To Do Beats Choosing the Right Tool

## Abstract

Retrieval-augmented systems typically commit to a single retrieval primitive — dense search, BM25, or graph traversal — regardless of query characteristics or operation cost. We study whether an adaptive routing policy over multiple retrieval substrates can improve cost-efficiency by selecting operations based on workspace coverage rather than query surface patterns.

We introduce a coverage-driven retrieval routing policy that operates over three heterogeneous substrates (semantic, lexical, entity-graph). The policy initiates with dense retrieval, evaluates whether the retrieved evidence is sufficient, and escalates to additional substrates only when coverage gaps are detected. Its primary mechanism is **selective stopping** — recognizing when current evidence suffices and avoiding unnecessary operations.

On HotpotQA Bridge (N=500, bootstrap CIs reported), the policy reduces average operations from 2.00 to 1.21 while maintaining comparable support recall (0.795 vs. 0.810 for BM25), yielding improved cost-efficiency. On a controlled heterogeneous benchmark (N=100, 6 task types with entity and lexical isolation), it matches the best single-substrate policy at fewer operations.

Ablation analysis reveals a counterintuitive finding: the policy's advantage derives from **routing avoidance** — knowing what *not* to do — rather than from positive substrate selection. Forcing unconditional escalation is catastrophic, while removing entity-graph hops slightly improves performance on lexically-rich data. We discuss the implications of this finding for the design of cost-efficient multi-substrate retrieval systems and identify learned routing as the key open challenge.
