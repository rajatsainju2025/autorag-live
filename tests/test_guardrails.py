"""Tests for guardrail enforcement layer."""


import pytest

from autorag_live.agent.guardrails import (
    EnforcementAction,
    GuardrailEnforcer,
    PermissionDenied,
    PermissionPolicy,
    RateLimitConfig,
    RateLimiter,
    RateLimitExceeded,
    ToolPermission,
)


class TestPermissionPolicy:
    def test_read_only(self):
        policy = PermissionPolicy.read_only()
        assert ToolPermission.READ in policy.allowed
        assert ToolPermission.NETWORK in policy.allowed
        assert ToolPermission.WRITE not in policy.allowed

    def test_permissive(self):
        policy = PermissionPolicy.permissive()
        assert len(policy.allowed) == len(ToolPermission)
        assert policy.enforcement == EnforcementAction.AUDIT

    def test_restricted(self):
        policy = PermissionPolicy.restricted()
        assert policy.allowed == {ToolPermission.READ}


class TestRateLimiter:
    def test_under_limit_passes(self):
        limiter = RateLimiter(RateLimitConfig(max_calls=5, window_seconds=60))
        for _ in range(5):
            limiter.check("tool_a")  # Should not raise

    def test_over_limit_raises(self):
        limiter = RateLimiter(RateLimitConfig(max_calls=2, window_seconds=60))
        limiter.check("tool_a")
        limiter.check("tool_a")
        with pytest.raises(RateLimitExceeded, match="tool_a"):
            limiter.check("tool_a")

    def test_per_tool_config(self):
        limiter = RateLimiter(RateLimitConfig(max_calls=100, window_seconds=60))
        limiter.configure_tool("expensive", RateLimitConfig(max_calls=1, window_seconds=60))

        limiter.check("cheap")  # uses default (100)
        limiter.check("expensive")
        with pytest.raises(RateLimitExceeded):
            limiter.check("expensive")

    def test_reset_tool(self):
        limiter = RateLimiter(RateLimitConfig(max_calls=1, window_seconds=60))
        limiter.check("tool_a")
        limiter.reset("tool_a")
        limiter.check("tool_a")  # Should not raise after reset

    def test_reset_all(self):
        limiter = RateLimiter(RateLimitConfig(max_calls=1, window_seconds=60))
        limiter.check("a")
        limiter.check("b")
        limiter.reset()
        limiter.check("a")
        limiter.check("b")


class TestGuardrailEnforcer:
    def test_allowed_passes(self):
        policy = PermissionPolicy(allowed={ToolPermission.READ, ToolPermission.NETWORK})
        enforcer = GuardrailEnforcer(policy)
        assert enforcer.check("web_search", {ToolPermission.READ, ToolPermission.NETWORK})

    def test_denied_raises(self):
        policy = PermissionPolicy(allowed={ToolPermission.READ})
        enforcer = GuardrailEnforcer(policy)
        with pytest.raises(PermissionDenied, match="write"):
            enforcer.check("delete_file", {ToolPermission.WRITE})

    def test_blocked_tool(self):
        policy = PermissionPolicy(allowed=set(ToolPermission), blocked_tools={"rm_rf"})
        enforcer = GuardrailEnforcer(policy)
        with pytest.raises(PermissionDenied):
            enforcer.check("rm_rf")

    def test_warn_mode_allows(self):
        policy = PermissionPolicy(
            allowed={ToolPermission.READ},
            enforcement=EnforcementAction.WARN,
        )
        enforcer = GuardrailEnforcer(policy)
        # Should not raise, just warn
        result = enforcer.check("tool", {ToolPermission.WRITE})
        assert result is True

    def test_audit_mode_allows(self):
        policy = PermissionPolicy.permissive()
        enforcer = GuardrailEnforcer(policy)
        enforcer.check("any_tool", {ToolPermission.ADMIN})
        assert len(enforcer.audit_log) == 0  # no violation

    def test_tool_overrides(self):
        policy = PermissionPolicy(
            allowed={ToolPermission.READ},
            tool_overrides={"special_tool": {ToolPermission.READ, ToolPermission.WRITE}},
        )
        enforcer = GuardrailEnforcer(policy)
        # special_tool has write override
        enforcer.check("special_tool", {ToolPermission.WRITE})
        # other tools don't
        with pytest.raises(PermissionDenied):
            enforcer.check("other_tool", {ToolPermission.WRITE})

    def test_audit_log_records_violations(self):
        policy = PermissionPolicy(
            allowed={ToolPermission.READ},
            enforcement=EnforcementAction.WARN,
        )
        enforcer = GuardrailEnforcer(policy)
        enforcer.check("tool", {ToolPermission.WRITE})
        assert len(enforcer.audit_log) == 1
        assert enforcer.audit_log[0]["tool"] == "tool"
        assert enforcer.audit_log[0]["reason"] == "permission_denied"

    def test_rate_limit_integration(self):
        policy = PermissionPolicy.permissive()
        limiter = RateLimiter(RateLimitConfig(max_calls=1, window_seconds=60))
        enforcer = GuardrailEnforcer(policy, limiter)

        enforcer.check("tool")
        with pytest.raises(RateLimitExceeded):
            enforcer.check("tool")

    def test_no_required_perms_passes(self):
        policy = PermissionPolicy(allowed={ToolPermission.READ})
        enforcer = GuardrailEnforcer(policy)
        assert enforcer.check("tool") is True
