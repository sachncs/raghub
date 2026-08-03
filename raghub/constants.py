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
