"""
Quick demo: convergence-retrieval in action.

Run: python examples/demo.py
"""

from convergence_retrieval import ConvergenceRetriever, BM25Substrate, DenseSubstrate, StructuralSubstrate

# A small document corpus (imagine these are code files, wiki articles, or enterprise docs)
documents = [
    {"id": "auth.py", "title": "Authentication Module", "content": "The authenticate() function validates JWT tokens by checking the signature against the secret key, verifying expiration timestamps, and extracting user claims. It raises AuthError for invalid or expired tokens."},
    {"id": "middleware.py", "title": "Middleware Stack", "content": "The auth_middleware checks each incoming HTTP request for a valid Bearer token in the Authorization header. If missing or invalid, it returns a 401 Unauthorized response before the request reaches the route handler."},
    {"id": "db.py", "title": "Database Layer", "content": "Connection pooling is managed by SQLAlchemy's create_engine with pool_size=10 and max_overflow=20. The get_session() context manager handles transaction commit/rollback automatically."},
    {"id": "cache.py", "title": "Cache System", "content": "Redis cache with 5-minute TTL for user sessions. The cache_user_profile decorator memoizes database lookups by user_id. Invalidation happens on profile update via pub/sub channel."},
    {"id": "api.py", "title": "API Endpoints", "content": "The /users endpoint supports GET (list all), POST (create), PUT (update), and DELETE (remove). All mutations require admin role. Rate limited to 100 requests per minute per API key."},
    {"id": "config.py", "title": "Configuration", "content": "Environment variables loaded from .env file via python-dotenv. DATABASE_URL, REDIS_URL, JWT_SECRET, and API_KEY_SALT are required. LOG_LEVEL defaults to INFO."},
    {"id": "tests.py", "title": "Test Suite", "content": "Pytest fixtures for database setup/teardown. Factory functions create test users with randomized attributes. Auth tests use a fixed JWT_SECRET for deterministic token generation."},
    {"id": "deploy.yaml", "title": "Deployment", "content": "Kubernetes deployment with 3 replicas, health check on /healthz, rolling update strategy. Secrets mounted from k8s Secret objects. HPA scales between 3-10 pods based on CPU utilization."},
    {"id": "logging.py", "title": "Logging", "content": "Structured JSON logging with correlation IDs propagated through middleware. Log levels: DEBUG for development, INFO for production. Logs shipped to Elasticsearch via Filebeat sidecar."},
    {"id": "models.py", "title": "Data Models", "content": "SQLAlchemy ORM models for User, Role, Permission, and AuditLog tables. User has a many-to-many relationship with Role through the user_roles association table."},
]

# Create retriever with 3 substrates
retriever = ConvergenceRetriever(
    substrates=[
        BM25Substrate(),        # keyword matching
        DenseSubstrate(),       # semantic similarity
        StructuralSubstrate(),  # title/path matching
    ],
)

# Index the documents
print("Indexing documents...")
retriever.index(documents)

# Run some queries
queries = [
    "How does JWT token validation work?",
    "What database connection pool settings are used?",
    "How are Kubernetes deployments configured?",
    "Where is the cache invalidation logic?",
    "What environment variables are required?",
]

print(f"\n{'='*70}")
print("CONVERGENCE RETRIEVAL DEMO")
print(f"{'='*70}")
print(f"Substrates: {retriever.substrate_names}")
print(f"Documents: {len(documents)}")
print()

for query in queries:
    result = retriever.search(query)
    top_doc = result.results[0] if result.results else None

    print(f"Q: {query}")
    print(f"  → Found: {top_doc.doc_id if top_doc else 'nothing'} (score={top_doc.score:.3f})" if top_doc else "  → No results")
    print(f"  → Ops: {result.ops_used}/{len(retriever.substrate_names)} "
          f"({'converged' if result.trace.converged else result.trace.stopped_reason}) "
          f"| Saved: {result.ops_saved} ops ({result.savings_pct:.0f}%)")
    print()

# Benchmark: convergence vs exhaustive
print(f"{'='*70}")
print("BENCHMARK: Convergence vs Exhaustive")
print(f"{'='*70}")
stats = retriever.benchmark(queries)
print(f"  Queries:            {stats['n_queries']}")
print(f"  Avg ops (converge): {stats['avg_ops_convergence']}")
print(f"  Avg ops (exhaust):  {stats['avg_ops_exhaustive']}")
print(f"  Savings:            {stats['savings_pct']}%")
print(f"  Convergence rate:   {stats['convergence_rate']}%")
print(f"  Result overlap:     {stats['avg_result_overlap']}%")
print(f"\n  → Convergence retrieval finds {stats['avg_result_overlap']:.0f}% of exhaustive results")
print(f"    at {stats['savings_pct']:.0f}% lower cost.")
