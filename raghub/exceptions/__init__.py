"""Custom exception hierarchy for the Dynamic RAG framework.

All package exceptions descend from :class:`DynamicRagError`, so
callers can catch every framework-raised error with a single
``except DynamicRagError`` block. Subclasses carry finer-grained
names so production handlers can react differently without inspecting
string messages.
"""

from __future__ import annotations


class DynamicRagError(Exception):
    """Base class for all package errors.

    Catch this to handle any framework-raised exception. Concrete
    subclasses provide the specific failure context.
    """


class AuthenticationError(DynamicRagError):
    """Raised when authentication fails (bad credentials, expired token)."""


class AuthorizationError(DynamicRagError):
    """Raised when a caller lacks permission for an action."""


class EmbeddingError(DynamicRagError):
    """Raised when text embedding fails (model error, dimension mismatch)."""


class RetrievalError(DynamicRagError):
    """Raised when retrieval fails (vector store, filter, RBAC)."""


class DocumentError(DynamicRagError):
    """Raised when document validation or lifecycle management fails."""


class IndexingError(DynamicRagError):
    """Raised when indexing or persistence fails."""


class PromptError(DynamicRagError):
    """Raised when prompt construction fails."""


class LLMError(DynamicRagError):
    """Raised when LLM generation fails."""


class StorageError(DynamicRagError):
    """Raised when durable storage fails."""
