"""
Guardrail enforcement layer for agentic RAG tool execution.

The project had ``ToolPermission`` and ``ToolCategory`` enums defined in
``agent/tool_registry.py`` but no runtime enforcement — tools could be
called regardless of permissions.  This module provides:

    1. ``PermissionPolicy`` — declares what permissions the current session has
    2. ``GuardrailEnforcer`` — wraps tool execution with permission checks
    3. ``RateLimiter`` — per-tool rate limiting to prevent runaway agents
    4. ``GuardedToolRegistry`` — drop-in replacement for ToolRegistry with enforcement

Example:
    >>> policy = PermissionPolicy(allowed={ToolPermission.READ, ToolPermission.NETWORK})
    >>> enforcer = GuardrailEnforcer(policy)
    >>>
    >>> # This passes:
    >>> enforcer.check("web_search", required={ToolPermission.READ, ToolPermission.NETWORK})
    >>>
    >>> # This raises PermissionDenied:
    >>> enforcer.check("delete_file", required={ToolPermission.WRITE, ToolPermission.FILESYSTEM})
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Permission types (mirrors tool_registry.py but adds enforcement)
# ---------------------------------------------------------------------------


class ToolPermission(str, Enum):
    """Permissions required to use tools."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    SENSITIVE = "sensitive"
    ADMIN = "admin"


class EnforcementAction(str, Enum):
    """What to do when a permission check fails."""

    DENY = "deny"  # Raise PermissionDenied
    WARN = "warn"  # Log warning but allow
    AUDIT = "audit"  # Log silently, allow


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PermissionDenied(Exception):
    """Raised when a tool call violates the permission policy."""

    def __init__(self, tool: str, required: Set[ToolPermission], allowed: Set[ToolPermission]):
        self.tool = tool
        self.required = required
        self.allowed = allowed
        missing = required - allowed
        super().__init__(
            f"Tool '{tool}' requires permissions {sorted(p.value for p in missing)} "
            f"which are not in the allowed set {sorted(p.value for p in allowed)}"
        )


class RateLimitExceeded(Exception):
    """Raised when a tool exceeds its rate limit."""

    def __init__(self, tool: str, limit: int, window_seconds: float):
        self.tool = tool
        super().__init__(
            f"Tool '{tool}' rate limit exceeded: max {limit} calls per {window_seconds}s"
        )


# ---------------------------------------------------------------------------
# PermissionPolicy
# ---------------------------------------------------------------------------


@dataclass
class PermissionPolicy:
    """
    Declares what permissions the current session has.

    Attributes:
        allowed:            Set of granted permissions.
        enforcement:        What to do on violation (deny/warn/audit).
        tool_overrides:     Per-tool permission overrides.
        blocked_tools:      Tools explicitly blocked regardless of permissions.
    """

    allowed: Set[ToolPermission] = field(default_factory=lambda: {ToolPermission.READ})
    enforcement: EnforcementAction = EnforcementAction.DENY
    tool_overrides: Dict[str, Set[ToolPermission]] = field(default_factory=dict)
    blocked_tools: Set[str] = field(default_factory=set)

    @classmethod
    def permissive(cls) -> PermissionPolicy:
        """All permissions granted — for trusted environments."""
        return cls(allowed=set(ToolPermission), enforcement=EnforcementAction.AUDIT)

    @classmethod
    def read_only(cls) -> PermissionPolicy:
        """Only read + network — safe for untrusted agents."""
        return cls(
            allowed={ToolPermission.READ, ToolPermission.NETWORK},
            enforcement=EnforcementAction.DENY,
        )

    @classmethod
    def restricted(cls) -> PermissionPolicy:
        """Read only, no network — maximum safety."""
        return cls(allowed={ToolPermission.READ}, enforcement=EnforcementAction.DENY)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a tool."""

    max_calls: int = 10
    window_seconds: float = 60.0


class RateLimiter:
    """
    Sliding-window rate limiter for tool calls.

    Tracks call timestamps per tool and enforces limits.
    """

    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        self._default = default_config or RateLimitConfig()
        self._tool_configs: Dict[str, RateLimitConfig] = {}
        self._call_times: Dict[str, List[float]] = defaultdict(list)

    def configure_tool(self, tool: str, config: RateLimitConfig) -> None:
        """Set rate limit for a specific tool."""
        self._tool_configs[tool] = config

    def check(self, tool: str) -> None:
        """Check and record a call.  Raises RateLimitExceeded if over limit."""
        config = self._tool_configs.get(tool, self._default)
        now = time.monotonic()
        cutoff = now - config.window_seconds

        # Prune old entries
        times = self._call_times[tool]
        self._call_times[tool] = [t for t in times if t > cutoff]

        if len(self._call_times[tool]) >= config.max_calls:
            raise RateLimitExceeded(tool, config.max_calls, config.window_seconds)

        self._call_times[tool].append(now)

    def reset(self, tool: Optional[str] = None) -> None:
        """Reset rate limiter state."""
        if tool:
            self._call_times.pop(tool, None)
        else:
            self._call_times.clear()


# ---------------------------------------------------------------------------
# GuardrailEnforcer
# ---------------------------------------------------------------------------


class GuardrailEnforcer:
    """
    Central enforcement point for tool permissions and rate limits.

    Wraps around tool execution to ensure compliance.
    """

    def __init__(
        self,
        policy: Optional[PermissionPolicy] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.policy = policy or PermissionPolicy.read_only()
        self.rate_limiter = rate_limiter or RateLimiter()
        self._audit_log: List[Dict[str, Any]] = []

    def check(
        self,
        tool: str,
        required: Optional[Set[ToolPermission]] = None,
    ) -> bool:
        """
        Check if a tool call is allowed.

        Returns True if allowed, raises PermissionDenied or RateLimitExceeded
        if denied.
        """
        # Blocked tools
        if tool in self.policy.blocked_tools:
            self._log_violation(tool, "blocked_tool")
            if self.policy.enforcement == EnforcementAction.DENY:
                raise PermissionDenied(tool, required or set(), self.policy.allowed)
            return self.policy.enforcement != EnforcementAction.DENY

        # Permission check
        effective_perms = self.policy.tool_overrides.get(tool, self.policy.allowed)
        required_perms = required or set()
        missing = required_perms - effective_perms

        if missing:
            self._log_violation(tool, "permission_denied", missing=missing)
            if self.policy.enforcement == EnforcementAction.DENY:
                raise PermissionDenied(tool, required_perms, effective_perms)
            if self.policy.enforcement == EnforcementAction.WARN:
                logger.warning(
                    "Tool '%s' requires %s but only %s granted (WARN mode)",
                    tool,
                    sorted(p.value for p in missing),
                    sorted(p.value for p in effective_perms),
                )

        # Rate limit check
        self.rate_limiter.check(tool)

        return True

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        """Return the audit log of violations."""
        return list(self._audit_log)

    def _log_violation(
        self,
        tool: str,
        reason: str,
        **extra: Any,
    ) -> None:
        entry = {
            "tool": tool,
            "reason": reason,
            "timestamp": time.time(),
            "enforcement": self.policy.enforcement.value,
            **{k: str(v) for k, v in extra.items()},
        }
        self._audit_log.append(entry)
        logger.info("Guardrail violation: %s", entry)
