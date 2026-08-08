"""Project-wide named constants.

Centralising magic numbers here satisfies AGENTS.md R8 (single source
of truth) and lets the linter / reviewers reject scattered literals.
"""

from __future__ import annotations

# Retrieval
RRF_K: int = 60

# Sessions
DEFAULT_SESSION_TIMEOUT_SECONDS: int = 3600

# Ingest
INGEST_WORKER_DEFAULT: int = 4

# HTTP / API
API_RATE_LIMIT_RPS: float = 10.0
API_RATE_LIMIT_BURST: float = 20.0

# Models
HASHING_BGE_MODEL: str = "hashing-bge"
GPT4O_MINI_MODEL: str = "gpt-4o-mini"
MINISHLAB_POTION_MODEL: str = "minishlab/potion-base-8M"

# Embeddings
DEFAULT_EMBEDDING_DIM: int = 384
DEFAULT_CHUNK_SIZE_WORDS: int = 800
DEFAULT_CHUNK_OVERLAP_WORDS: int = 100
DEFAULT_TOP_K: int = 5

# Prompts
DEFAULT_PROMPT_MAX_TOKENS: int = 4096
DEFAULT_PROMPT_RESERVED_TOKENS: int = 512

# Tools / web search
WEB_SEARCH_MAX_RESULTS: int = 25

# Uploads
DEFAULT_MAX_UPLOAD_BYTES: int = 20 * 1024 * 1024

# Queue
MAX_INFLIGHT_DEFAULT: int = 256

# Rate limit
RATE_LIMIT_RPS: float = 10.0
RATE_LIMIT_BURST: int = 20
RATE_LIMIT_USER_RPS: float = 5.0
RATE_LIMIT_USER_BURST: int = 10

# Archive
DEFAULT_ARCHIVE_DIR = "./data/archives"

# SQLite
SQLITE_BUSY_TIMEOUT_MS: int = 5000

# HTTP status codes (centralised so routing handlers stay declarative)
HTTP_400_BAD_REQUEST: int = 400
HTTP_401_UNAUTHORIZED: int = 401
HTTP_403_FORBIDDEN: int = 403
HTTP_404_NOT_FOUND: int = 404
HTTP_413_PAYLOAD_TOO_LARGE: int = 413
HTTP_422_UNPROCESSABLE: int = 422
HTTP_429_TOO_MANY_REQUESTS: int = 429
HTTP_500_INTERNAL_SERVER_ERROR: int = 500
HTTP_503_SERVICE_UNAVAILABLE: int = 503
