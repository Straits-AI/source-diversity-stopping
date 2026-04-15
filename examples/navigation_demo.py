"""
Navigation Agent Demo — this is NOT RAG.

RAG:        query → retrieve chunks → stuff into prompt → generate
Navigation: query → explore → decide → explore more → decide → stop

The agent navigates an information environment, maintaining what it has
DISCOVERED (knows exists) vs what it KNOWS (has actually read).
"""

import sys
sys.path.insert(0, ".")

from convergence_retrieval.substrates import BM25Substrate, DenseSubstrate, StructuralSubstrate
from convergence_retrieval.environments import DocumentEnvironment
from convergence_retrieval.navigation import NavigationAgent

# A small codebase-like environment
documents = [
    {
        "id": "auth.py",
        "title": "Authentication Module",
        "content": "The authenticate() function validates JWT tokens by checking the signature "
                   "against the secret key stored in config.JWT_SECRET. It uses jwt_utils.decode_token() "
                   "for the actual decoding. Invalid tokens raise AuthError. See also: middleware.py",
        "links": ["jwt_utils.py", "middleware.py"],
    },
    {
        "id": "jwt_utils.py",
        "title": "JWT Utilities",
        "content": "decode_token(token, secret) verifies the JWT signature and extracts claims. "
                   "encode_token(payload, secret, ttl=3600) creates a new signed JWT with expiration. "
                   "Both functions use the PyJWT library internally.",
    },
    {
        "id": "middleware.py",
        "title": "Request Middleware",
        "content": "auth_middleware intercepts all incoming HTTP requests. It extracts the Bearer "
                   "token from the Authorization header and calls auth.authenticate() to validate. "
                   "Returns 401 if the token is missing or invalid. Also applies rate_limiter().",
        "links": ["auth.py"],
    },
    {
        "id": "config.py",
        "title": "Configuration",
        "content": "Application settings loaded from environment variables: DATABASE_URL, REDIS_URL, "
                   "JWT_SECRET (required for token signing/verification), API_KEY_SALT, LOG_LEVEL (default: INFO).",
    },
    {
        "id": "models.py",
        "title": "Database Models",
        "content": "SQLAlchemy ORM models: User (id, email, password_hash, role), "
                   "Session (id, user_id, token, expires_at), Permission (id, name, description). "
                   "User has many Sessions. User has many Permissions through user_permissions table.",
    },
    {
        "id": "api.py",
        "title": "API Endpoints",
        "content": "REST endpoints: POST /login (authenticate and return JWT), GET /profile (requires auth), "
                   "PUT /profile (update user data), DELETE /session (logout). All protected endpoints "
                   "pass through auth_middleware.",
        "links": ["middleware.py"],
    },
    {
        "id": "tests/test_auth.py",
        "title": "Auth Tests",
        "content": "Test cases for JWT validation: test_valid_token, test_expired_token, "
                   "test_invalid_signature, test_missing_claims. Uses a fixed JWT_SECRET='test-secret' "
                   "for deterministic token generation. Fixtures create test users.",
    },
    {
        "id": "deploy.yaml",
        "title": "Deployment Config",
        "content": "Kubernetes deployment: 3 replicas, health check on /healthz, "
                   "JWT_SECRET injected from k8s Secret. Rolling update with maxSurge=1.",
    },
]

# Build environment with 3 substrates
env = DocumentEnvironment(
    substrates=[BM25Substrate(), DenseSubstrate(), StructuralSubstrate()]
)
env.load(documents)

# Create navigation agent
agent = NavigationAgent(environment=env, max_steps=8)

# Navigate!
print("=" * 70)
print("NAVIGATION AGENT DEMO")
print("=" * 70)
print()

queries = [
    "How does JWT token validation work in this codebase?",
    "What environment variables does the application need?",
    "How is the auth middleware connected to the API endpoints?",
]

for query in queries:
    print(f"{'─' * 70}")
    result = agent.navigate(query)
    print(result.trace.show())
    print(f"  📊 Discovered {result.discovered} items, read {result.read} items, used {result.ops_used} ops")
    print()

# Compare: navigation vs exhaustive retrieval
print("=" * 70)
print("NAVIGATION vs EXHAUSTIVE COMPARISON")
print("=" * 70)

from convergence_retrieval import ConvergenceRetriever

retriever = ConvergenceRetriever(
    substrates=[BM25Substrate(), DenseSubstrate(), StructuralSubstrate()]
)
retriever.index(documents)

for query in queries:
    nav_result = agent.navigate(query)
    exh_result = retriever.search_exhaustive(query)

    nav_ids = {k["doc_id"] for k in nav_result.knowledge}
    exh_ids = {r.doc_id for r in exh_result.results[:5]}

    overlap = nav_ids & exh_ids
    nav_extra = nav_ids - exh_ids  # things navigation found that retrieval missed

    print(f"\nQ: {query[:60]}...")
    print(f"  Navigation: {nav_result.ops_used} ops → {nav_ids}")
    print(f"  Exhaustive: {exh_result.ops_used} ops → {exh_ids}")
    if nav_extra:
        print(f"  Navigation found EXTRA (via link-following): {nav_extra}")
    print(f"  Overlap: {len(overlap)}/{len(exh_ids)}")
