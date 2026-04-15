"""End-to-end tests for ConvergenceRetriever."""

from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate, StructuralSubstrate


def make_docs():
    return [
        {"id": "auth", "title": "Authentication Module", "content": "The authentication module validates JWT tokens and checks user permissions against the role database."},
        {"id": "db", "title": "Database Layer", "content": "The database layer provides connection pooling and ORM mapping for PostgreSQL and MySQL backends."},
        {"id": "api", "title": "API Endpoints", "content": "REST API endpoints handle HTTP requests, validate input schemas, and return JSON responses."},
        {"id": "cache", "title": "Cache System", "content": "Redis-based caching layer with TTL expiration and LRU eviction for frequently accessed data."},
        {"id": "logging", "title": "Logging Framework", "content": "Structured logging with JSON output, log levels, and integration with ELK stack for monitoring."},
        {"id": "auth_test", "title": "Auth Tests", "content": "Unit tests for JWT validation, token expiration, and role-based access control assertions."},
        {"id": "deploy", "title": "Deployment Config", "content": "Docker compose files and Kubernetes manifests for staging and production environments."},
        {"id": "middleware", "title": "Middleware Stack", "content": "Request middleware for rate limiting, CORS headers, request ID injection, and authentication checks."},
    ]


def test_basic_search():
    retriever = ConvergenceRetriever(
        substrates=[BM25Substrate(), DenseSubstrate()],
    )
    retriever.index(make_docs())
    result = retriever.search("How does JWT token validation work?")

    assert len(result.results) > 0
    assert result.ops_used <= 2
    assert result.trace.query == "How does JWT token validation work?"
    # Should find auth-related docs
    result_ids = {r.doc_id for r in result.results}
    assert "auth" in result_ids or "auth_test" in result_ids or "middleware" in result_ids


def test_convergence_stops_early():
    retriever = ConvergenceRetriever(
        substrates=[BM25Substrate(), DenseSubstrate(), StructuralSubstrate()],
    )
    retriever.index(make_docs())
    result = retriever.search("authentication module JWT tokens")

    # With 3 substrates, convergence should stop before using all 3
    # (BM25 and Dense should both find auth docs → converge after 2)
    assert result.ops_used <= 2
    assert result.trace.converged
    assert result.ops_saved >= 1


def test_exhaustive_uses_all():
    retriever = ConvergenceRetriever(
        substrates=[BM25Substrate(), DenseSubstrate(), StructuralSubstrate()],
    )
    retriever.index(make_docs())
    result = retriever.search_exhaustive("authentication")

    assert result.ops_used == 3
    assert result.ops_saved == 0
    assert result.trace.stopped_reason == "exhaustive"


def test_benchmark():
    retriever = ConvergenceRetriever(
        substrates=[BM25Substrate(), DenseSubstrate()],
    )
    retriever.index(make_docs())

    queries = [
        "JWT token validation",
        "database connection pooling",
        "REST API endpoints",
        "Redis caching TTL",
        "Docker deployment",
    ]
    stats = retriever.benchmark(queries)

    assert stats["n_queries"] == 5
    assert stats["avg_ops_convergence"] <= stats["avg_ops_exhaustive"]
    assert stats["savings_pct"] >= 0
    assert stats["avg_result_overlap"] > 0


def test_min_two_substrates():
    try:
        ConvergenceRetriever(substrates=[BM25Substrate()])
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "at least 2" in str(e)


def test_index_required():
    retriever = ConvergenceRetriever(
        substrates=[BM25Substrate(), DenseSubstrate()],
    )
    try:
        retriever.search("test")
        assert False, "Should raise RuntimeError"
    except RuntimeError as e:
        assert "index" in str(e).lower()


if __name__ == "__main__":
    print("Running tests...")
    test_basic_search()
    print("  ✓ test_basic_search")
    test_convergence_stops_early()
    print("  ✓ test_convergence_stops_early")
    test_exhaustive_uses_all()
    print("  ✓ test_exhaustive_uses_all")
    test_benchmark()
    print("  ✓ test_benchmark")
    test_min_two_substrates()
    print("  ✓ test_min_two_substrates")
    test_index_required()
    print("  ✓ test_index_required")
    print("\nAll tests passed!")
