"""
Error recovery and iterative refinement for agent operations.

Implements retry logic, reflection-based refinement, and fallback chains.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple


class RecoveryStrategy(Enum):
    """Recovery strategies for failed operations."""

    RETRY = "retry"  # Retry with backoff
    FALLBACK = "fallback"  # Try fallback action
    REFINE = "refine"  # Refine and retry
    SKIP = "skip"  # Skip this step
    ABORT = "abort"  # Abort entire operation


@dataclass
class RetryConfig:
    """Configuration for retry logic."""

    max_retries: int = 3
    initial_backoff_ms: float = 100.0
    max_backoff_ms: float = 10000.0
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1  # Random jitter (0-10%)

    def get_backoff_delay(self, retry_num: int) -> float:
        """Calculate backoff delay for retry."""
        import random

        delay = self.initial_backoff_ms * (self.backoff_multiplier**retry_num)
        delay = min(delay, self.max_backoff_ms)

        # Add jitter
        jitter = delay * self.jitter_factor * random.random()
        return (delay + jitter) / 1000.0


@dataclass
class RecoveryAttempt:
    """Single recovery attempt."""

    attempt_number: int
    strategy: RecoveryStrategy
    timestamp: float = field(default_factory=time.time)
    description: str = ""
    success: bool = False
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Export attempt as dictionary."""
        return {
            "attempt": self.attempt_number,
            "strategy": self.strategy.value,
            "description": self.description,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


class ErrorRecoveryEngine:
    """
    Handles error recovery for agent operations.

    Implements retry logic, fallback chains, and reflection-based refinement.
    """

    def __init__(self):
        """Initialize error recovery engine."""
        self.recovery_history: List[RecoveryAttempt] = []
        self.logger = logging.getLogger("ErrorRecoveryEngine")

    def retry_with_backoff(
        self,
        operation: Callable,
        config: Optional[RetryConfig] = None,
        context: Optional[dict] = None,
    ) -> Tuple[Any, bool]:
        """
        Retry operation with exponential backoff.

        Args:
            operation: Callable to retry
            config: Retry configuration
            context: Optional context

        Returns:
            (result, success)
        """
        config = config or RetryConfig()
        context = context or {}

        for retry_num in range(config.max_retries):
            try:
                result = operation()
                if retry_num > 0:
                    attempt = RecoveryAttempt(
                        attempt_number=retry_num,
                        strategy=RecoveryStrategy.RETRY,
                        description=f"Retry succeeded on attempt {retry_num + 1}",
                        success=True,
                    )
                    self.recovery_history.append(attempt)
                return result, True

            except Exception as e:
                if retry_num < config.max_retries - 1:
                    backoff = config.get_backoff_delay(retry_num)
                    self.logger.warning(
                        f"Operation failed (attempt {retry_num + 1}), retrying in {backoff:.1f}s: {str(e)}"
                    )
                    time.sleep(backoff)
                else:
                    attempt = RecoveryAttempt(
                        attempt_number=retry_num,
                        strategy=RecoveryStrategy.RETRY,
                        description=f"All {config.max_retries} retries exhausted",
                        success=False,
                        error=str(e),
                    )
                    self.recovery_history.append(attempt)
                    self.logger.error(
                        f"Operation failed after {config.max_retries} retries: {str(e)}"
                    )
                    return None, False

        return None, False

    def try_fallback_chain(
        self, operations: List[Tuple[str, Callable]]
    ) -> Tuple[Optional[Any], str, bool]:
        """
        Try chain of fallback operations.

        Args:
            operations: List of (name, operation) tuples

        Returns:
            (result, operation_name, success)
        """
        for op_name, operation in operations:
            try:
                self.logger.info(f"Trying fallback: {op_name}")
                result = operation()

                attempt = RecoveryAttempt(
                    attempt_number=len(self.recovery_history),
                    strategy=RecoveryStrategy.FALLBACK,
                    description=f"Fallback succeeded: {op_name}",
                    success=True,
                    metadata={"fallback_operation": op_name},
                )
                self.recovery_history.append(attempt)

                return result, op_name, True

            except Exception as e:
                self.logger.debug(f"Fallback {op_name} failed: {str(e)}")
                continue

        attempt = RecoveryAttempt(
            attempt_number=len(self.recovery_history),
            strategy=RecoveryStrategy.FALLBACK,
            description="All fallback operations exhausted",
            success=False,
            error="No fallback operations succeeded",
        )
        self.recovery_history.append(attempt)

        return None, "", False


class ReflectionEngine:
    """Reflects on failures and suggests refinements."""

    def __init__(self):
        """Initialize reflection engine."""
        self.logger = logging.getLogger("ReflectionEngine")

    def analyze_failure(self, error: str, context: Optional[dict] = None) -> dict:
        """
        Analyze failure and suggest recovery.

        Args:
            error: Error message
            context: Optional context

        Returns:
            Analysis dictionary with suggestions
        """
        context = context or {}

        # Simple error pattern matching
        analysis = {
            "error": error,
            "error_type": self._classify_error(error),
            "suggested_strategies": [],
            "suggested_refinements": [],
        }

        if "timeout" in error.lower():
            analysis["error_type"] = "timeout"
            analysis["suggested_strategies"] = [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.REFINE,
                RecoveryStrategy.FALLBACK,
            ]
            analysis["suggested_refinements"].append("Increase timeout threshold")
            analysis["suggested_refinements"].append("Simplify operation")

        elif "not found" in error.lower():
            analysis["error_type"] = "not_found"
            analysis["suggested_strategies"] = [RecoveryStrategy.REFINE, RecoveryStrategy.FALLBACK]
            analysis["suggested_refinements"].append("Broaden search criteria")
            analysis["suggested_refinements"].append("Use different tool")

        elif "invalid" in error.lower():
            analysis["error_type"] = "invalid_input"
            analysis["suggested_strategies"] = [RecoveryStrategy.REFINE]
            analysis["suggested_refinements"].append("Validate inputs")
            analysis["suggested_refinements"].append("Preprocess data")

        else:
            analysis["error_type"] = "unknown"
            analysis["suggested_strategies"] = [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK]

        return analysis

    def _classify_error(self, error: str) -> str:
        """Classify error type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "not found" in error_lower:
            return "not_found"
        elif "invalid" in error_lower:
            return "invalid_input"
        elif "permission" in error_lower or "forbidden" in error_lower:
            return "permission_denied"
        elif "connection" in error_lower or "network" in error_lower:
            return "network_error"
        else:
            return "unknown"

    def suggest_refinement(self, original_input: str, error: str) -> str:
        """
        Suggest refined version of input.

        Args:
            original_input: Original input
            error: Error that occurred

        Returns:
            Suggested refinement
        """
        analysis = self.analyze_failure(error)

        if analysis["error_type"] == "not_found":
            # Broaden search
            return f"{original_input} OR alternative OR similar"

        elif analysis["error_type"] == "invalid_input":
            # Simplify
            words = original_input.split()
            if len(words) > 3:
                return " ".join(words[:3])
            return original_input

        else:
            return original_input


class AdaptiveExecutor:
    """
    Adaptively executes operations with built-in recovery.

    Combines retry logic, fallbacks, and reflection.
    """

    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        enable_reflection: bool = True,
    ):
        """
        Initialize adaptive executor.

        Args:
            retry_config: Retry configuration
            enable_reflection: Enable reflection-based refinement
        """
        self.retry_config = retry_config or RetryConfig()
        self.recovery_engine = ErrorRecoveryEngine()
        self.reflection_engine = ReflectionEngine() if enable_reflection else None
        self.logger = logging.getLogger("AdaptiveExecutor")

    def execute(
        self,
        operation: Callable,
        fallbacks: Optional[List[Tuple[str, Callable]]] = None,
        context: Optional[dict] = None,
    ) -> Tuple[Optional[Any], str]:
        """
        Execute operation with full recovery pipeline.

        Args:
            operation: Main operation
            fallbacks: Optional fallback operations
            context: Optional context

        Returns:
            (result, execution_status)
        """
        context = context or {}

        # Try main operation with retry
        result, success = self.recovery_engine.retry_with_backoff(
            operation, self.retry_config, context
        )

        if success:
            return result, "success"

        # If main operation failed and we have fallbacks, try them
        if fallbacks:
            self.logger.info("Trying fallback operations")
            result, fallback_name, success = self.recovery_engine.try_fallback_chain(fallbacks)

            if success:
                return result, f"fallback:{fallback_name}"

        return None, "failed"

    def get_recovery_report(self) -> dict:
        """Get report of all recovery attempts."""
        return {
            "total_attempts": len(self.recovery_engine.recovery_history),
            "attempts": [a.to_dict() for a in self.recovery_engine.recovery_history],
        }


class ResilientAgentWrapper:
    """Wraps agent with resilience layer."""

    def __init__(self, agent, retry_config: Optional[RetryConfig] = None):
        """
        Initialize resilient agent wrapper.

        Args:
            agent: Base agent
            retry_config: Optional retry configuration
        """
        self.agent = agent
        self.executor = AdaptiveExecutor(retry_config=retry_config)
        self.logger = logging.getLogger("ResilientAgentWrapper")

    def execute_action_resilient(
        self, action: Any, fallback_actions: Optional[List[Any]] = None
    ) -> Tuple[Any, str]:
        """
        Execute action with resilience.

        Args:
            action: Action to execute
            fallback_actions: Optional fallback actions

        Returns:
            (observation, status)
        """

        def main_operation():
            return self.agent.act(action)

        fallbacks = None
        if fallback_actions:
            fallbacks = [
                (f"fallback_{i}", lambda a=fa: self.agent.act(a))
                for i, fa in enumerate(fallback_actions)
            ]

        result, status = self.executor.execute(main_operation, fallbacks)
        return result, status

    def get_resilience_report(self) -> dict:
        """Get resilience report."""
        return {
            "recovery_attempts": self.executor.get_recovery_report(),
            "agent_state": self.agent.get_state_summary(),
        }
