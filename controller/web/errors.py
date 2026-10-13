from __future__ import annotations


class RuntimeConflict(RuntimeError):
    """The requested control action conflicts with current runtime state."""


class RuntimeUnavailable(RuntimeError):
    """The requested control action cannot currently be performed."""
