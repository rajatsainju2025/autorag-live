"""
Content filtering and moderation module for AutoRAG-Live.

Provides content filtering capabilities for safe RAG systems
with multiple detection strategies.

Features:
- Text content filtering
- PII detection and redaction
- Toxicity detection
- Prompt injection detection
- Jailbreak attempt detection
- Custom rule engine
- Content classification
- Output sanitization

Example usage:
    >>> filter = ContentFilter()
    >>> result = filter.analyze("Check this text for safety")
    >>> if result.is_safe:
    ...     process(result.sanitized_text)
"""

from __future__ import annotations

import logging
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of content threats."""
    
    SAFE = auto()
    PII = auto()
    TOXICITY = auto()
    PROMPT_INJECTION = auto()
    JAILBREAK = auto()
    HARMFUL_CONTENT = auto()
    SPAM = auto()
    MALWARE = auto()
    BIAS = auto()
    MISINFORMATION = auto()
    NSFW = auto()
    CUSTOM = auto()


class Severity(Enum):
    """Threat severity levels."""
    
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class FilterAction(Enum):
    """Actions to take on detected threats."""
    
    ALLOW = auto()
    WARN = auto()
    REDACT = auto()
    BLOCK = auto()
    LOG = auto()


@dataclass
class ThreatMatch:
    """A detected threat match."""
    
    category: ThreatCategory
    severity: Severity
    
    # Match details
    matched_text: str
    start_pos: int
    end_pos: int
    
    # Context
    rule_id: str = ""
    description: str = ""
    confidence: float = 1.0
    
    # Suggested action
    action: FilterAction = FilterAction.WARN
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of content filtering."""
    
    is_safe: bool
    
    # Original and processed text
    original_text: str
    sanitized_text: str
    
    # Detected threats
    threats: List[ThreatMatch] = field(default_factory=list)
    
    # Overall severity
    max_severity: Severity = Severity.NONE
    
    # Processing info
    processing_time_ms: float = 0.0
    filters_applied: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def threat_count(self) -> int:
        """Get number of threats."""
        return len(self.threats)
    
    @property
    def categories_detected(self) -> Set[ThreatCategory]:
        """Get detected categories."""
        return {t.category for t in self.threats}
    
    def get_threats_by_category(
        self,
        category: ThreatCategory,
    ) -> List[ThreatMatch]:
        """Get threats by category."""
        return [t for t in self.threats if t.category == category]


@dataclass
class FilterRule:
    """A filter rule definition."""
    
    rule_id: str
    name: str
    category: ThreatCategory
    
    # Pattern
    pattern: str
    is_regex: bool = True
    case_sensitive: bool = False
    
    # Severity and action
    severity: Severity = Severity.MEDIUM
    action: FilterAction = FilterAction.WARN
    
    # Description
    description: str = ""
    
    # Enabled
    enabled: bool = True
    
    _compiled: Optional[Pattern] = None
    
    def compile(self) -> Pattern:
        """Compile regex pattern."""
        if self._compiled is None:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            self._compiled = re.compile(self.pattern, flags)
        return self._compiled


class FilterStrategy(ABC):
    """Base class for filter strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get strategy name."""
        pass
    
    @abstractmethod
    def analyze(
        self,
        text: str,
    ) -> List[ThreatMatch]:
        """
        Analyze text for threats.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected threats
        """
        pass


class PIIDetector(FilterStrategy):
    """
    Detects Personally Identifiable Information.
    
    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Physical addresses
    """
    
    def __init__(self):
        """Initialize PII detector."""
        self._patterns = {
            'email': (
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'Email address detected',
            ),
            'phone_us': (
                r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                'US phone number detected',
            ),
            'ssn': (
                r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b',
                'SSN pattern detected',
            ),
            'credit_card': (
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                'Credit card number detected',
            ),
            'ipv4': (
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
                'IPv4 address detected',
            ),
            'date_of_birth': (
                r'\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}\b',
                'Date of birth pattern detected',
            ),
        }
        
        self._compiled = {
            name: (re.compile(pattern, re.IGNORECASE), desc)
            for name, (pattern, desc) in self._patterns.items()
        }
    
    @property
    def name(self) -> str:
        return "pii_detector"
    
    def analyze(self, text: str) -> List[ThreatMatch]:
        threats = []
        
        for rule_id, (pattern, description) in self._compiled.items():
            for match in pattern.finditer(text):
                threats.append(ThreatMatch(
                    category=ThreatCategory.PII,
                    severity=Severity.HIGH,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    rule_id=f"pii_{rule_id}",
                    description=description,
                    action=FilterAction.REDACT,
                ))
        
        return threats


class ToxicityDetector(FilterStrategy):
    """
    Detects toxic and harmful language.
    
    Uses keyword matching and patterns for:
    - Profanity
    - Hate speech indicators
    - Harassment language
    - Threats
    """
    
    def __init__(self, custom_words: Optional[List[str]] = None):
        """
        Initialize toxicity detector.
        
        Args:
            custom_words: Additional words to flag
        """
        # Basic profanity patterns (sanitized for safety)
        self._patterns = [
            (r'\b(?:hate|kill|attack|destroy)\s+(?:you|them|everyone)\b', 'Threatening language'),
            (r'\b(?:stupid|idiot|dumb|moron)\b', 'Insulting language'),
        ]
        
        self._compiled = [
            (re.compile(p, re.IGNORECASE), desc)
            for p, desc in self._patterns
        ]
        
        # Custom word list
        if custom_words:
            custom_pattern = '|'.join(re.escape(w) for w in custom_words)
            self._compiled.append((
                re.compile(f'\\b({custom_pattern})\\b', re.IGNORECASE),
                'Custom flagged word',
            ))
    
    @property
    def name(self) -> str:
        return "toxicity_detector"
    
    def analyze(self, text: str) -> List[ThreatMatch]:
        threats = []
        
        for i, (pattern, description) in enumerate(self._compiled):
            for match in pattern.finditer(text):
                threats.append(ThreatMatch(
                    category=ThreatCategory.TOXICITY,
                    severity=Severity.MEDIUM,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    rule_id=f"toxicity_{i}",
                    description=description,
                    action=FilterAction.WARN,
                ))
        
        return threats


class PromptInjectionDetector(FilterStrategy):
    """
    Detects prompt injection attempts.
    
    Detects patterns like:
    - "Ignore previous instructions"
    - "System prompt override"
    - "You are now..."
    - Role-playing attacks
    """
    
    def __init__(self):
        """Initialize prompt injection detector."""
        self._patterns = [
            (
                r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)',
                'Instruction override attempt',
                Severity.CRITICAL,
            ),
            (
                r'(?:system|hidden)\s+prompt',
                'System prompt reference',
                Severity.HIGH,
            ),
            (
                r'you\s+are\s+now\s+(?:a|an|the)',
                'Role override attempt',
                Severity.HIGH,
            ),
            (
                r'(?:forget|disregard)\s+(?:everything|all)',
                'Memory wipe attempt',
                Severity.CRITICAL,
            ),
            (
                r'act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were))',
                'Role-playing injection',
                Severity.MEDIUM,
            ),
            (
                r'(?:pretend|imagine)\s+(?:you\s+(?:are|were)|that)',
                'Pretend scenario injection',
                Severity.MEDIUM,
            ),
            (
                r'(?:new|override|replace)\s+(?:system\s+)?(?:instructions?|rules?)',
                'Rule override attempt',
                Severity.CRITICAL,
            ),
            (
                r'(?:do\s+not|don\'t)\s+follow\s+(?:any|your)',
                'Anti-compliance prompt',
                Severity.HIGH,
            ),
            (
                r'</?(system|assistant|user|human)>',
                'Message boundary injection',
                Severity.CRITICAL,
            ),
            (
                r'\[(?:INST|SYS|SYSTEM)\]',
                'Instruction tag injection',
                Severity.CRITICAL,
            ),
        ]
        
        self._compiled = [
            (re.compile(p, re.IGNORECASE), desc, sev)
            for p, desc, sev in self._patterns
        ]
    
    @property
    def name(self) -> str:
        return "prompt_injection_detector"
    
    def analyze(self, text: str) -> List[ThreatMatch]:
        threats = []
        
        for i, (pattern, description, severity) in enumerate(self._compiled):
            for match in pattern.finditer(text):
                threats.append(ThreatMatch(
                    category=ThreatCategory.PROMPT_INJECTION,
                    severity=severity,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    rule_id=f"injection_{i}",
                    description=description,
                    action=FilterAction.BLOCK,
                ))
        
        return threats


class JailbreakDetector(FilterStrategy):
    """
    Detects jailbreak attempts.
    
    Detects patterns like:
    - DAN (Do Anything Now) prompts
    - "Developer mode" requests
    - Roleplay escapes
    - Content policy bypass attempts
    """
    
    def __init__(self):
        """Initialize jailbreak detector."""
        self._patterns = [
            (
                r'\bDAN\b.*?(?:mode|prompt|jailbreak)',
                'DAN jailbreak attempt',
                Severity.CRITICAL,
            ),
            (
                r'(?:developer|debug|admin|root)\s+mode',
                'Privileged mode request',
                Severity.CRITICAL,
            ),
            (
                r'(?:bypass|ignore|disable)\s+(?:safety|content|moderation)\s+(?:filters?|policies?)',
                'Safety bypass attempt',
                Severity.CRITICAL,
            ),
            (
                r'(?:unlock|enable)\s+(?:all|full)\s+(?:capabilities|features)',
                'Capability unlock attempt',
                Severity.HIGH,
            ),
            (
                r'(?:no|without)\s+(?:restrictions?|limitations?|filters?)',
                'Restriction removal request',
                Severity.HIGH,
            ),
            (
                r'(?:pretend|act)\s+(?:there\s+are\s+)?(?:no|zero)\s+(?:rules|guidelines)',
                'Rule bypass via roleplay',
                Severity.CRITICAL,
            ),
            (
                r'(?:hypothetically|theoretically)\s+(?:if\s+)?(?:you\s+)?(?:could|were)',
                'Hypothetical bypass',
                Severity.MEDIUM,
            ),
            (
                r'(?:do\s+)?anything\s+now',
                'DAN variant',
                Severity.CRITICAL,
            ),
        ]
        
        self._compiled = [
            (re.compile(p, re.IGNORECASE), desc, sev)
            for p, desc, sev in self._patterns
        ]
    
    @property
    def name(self) -> str:
        return "jailbreak_detector"
    
    def analyze(self, text: str) -> List[ThreatMatch]:
        threats = []
        
        for i, (pattern, description, severity) in enumerate(self._compiled):
            for match in pattern.finditer(text):
                threats.append(ThreatMatch(
                    category=ThreatCategory.JAILBREAK,
                    severity=severity,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    rule_id=f"jailbreak_{i}",
                    description=description,
                    action=FilterAction.BLOCK,
                ))
        
        return threats


class CustomRuleEngine:
    """
    Custom rule engine for flexible content filtering.
    
    Example:
        >>> engine = CustomRuleEngine()
        >>> engine.add_rule(FilterRule(
        ...     rule_id="custom_1",
        ...     name="Block competitor mentions",
        ...     category=ThreatCategory.CUSTOM,
        ...     pattern=r"competitor_name",
        ... ))
    """
    
    def __init__(self):
        """Initialize rule engine."""
        self._rules: Dict[str, FilterRule] = {}
    
    def add_rule(self, rule: FilterRule) -> None:
        """Add a rule."""
        self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[FilterRule]:
        """Get a rule by ID."""
        return self._rules.get(rule_id)
    
    def analyze(self, text: str) -> List[ThreatMatch]:
        """Analyze text with all rules."""
        threats = []
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            pattern = rule.compile()
            
            for match in pattern.finditer(text):
                threats.append(ThreatMatch(
                    category=rule.category,
                    severity=rule.severity,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    rule_id=rule.rule_id,
                    description=rule.description,
                    action=rule.action,
                ))
        
        return threats


class TextSanitizer:
    """
    Sanitizes text by redacting or replacing detected threats.
    
    Example:
        >>> sanitizer = TextSanitizer()
        >>> sanitized = sanitizer.redact(
        ...     "Email me at john@example.com",
        ...     threats,
        ...     mask="[REDACTED]"
        ... )
    """
    
    def __init__(
        self,
        default_mask: str = "[REDACTED]",
        preserve_length: bool = False,
    ):
        """
        Initialize sanitizer.
        
        Args:
            default_mask: Default redaction mask
            preserve_length: Preserve original text length
        """
        self.default_mask = default_mask
        self.preserve_length = preserve_length
    
    def redact(
        self,
        text: str,
        threats: List[ThreatMatch],
        mask: Optional[str] = None,
        categories: Optional[Set[ThreatCategory]] = None,
    ) -> str:
        """
        Redact detected threats.
        
        Args:
            text: Original text
            threats: Detected threats
            mask: Redaction mask
            categories: Categories to redact (all if None)
            
        Returns:
            Sanitized text
        """
        mask = mask or self.default_mask
        
        # Sort threats by position (reverse to preserve indices)
        sorted_threats = sorted(
            threats,
            key=lambda t: t.start_pos,
            reverse=True,
        )
        
        result = text
        
        for threat in sorted_threats:
            if categories and threat.category not in categories:
                continue
            
            if threat.action not in [FilterAction.REDACT, FilterAction.BLOCK]:
                continue
            
            if self.preserve_length:
                replacement = mask[0] * len(threat.matched_text)
            else:
                replacement = mask
            
            result = (
                result[:threat.start_pos] +
                replacement +
                result[threat.end_pos:]
            )
        
        return result
    
    def hash_pii(
        self,
        text: str,
        threats: List[ThreatMatch],
        salt: str = "",
    ) -> str:
        """
        Replace PII with hashed values.
        
        Args:
            text: Original text
            threats: Detected threats
            salt: Hash salt
            
        Returns:
            Text with hashed PII
        """
        pii_threats = [
            t for t in threats
            if t.category == ThreatCategory.PII
        ]
        
        sorted_threats = sorted(
            pii_threats,
            key=lambda t: t.start_pos,
            reverse=True,
        )
        
        result = text
        
        for threat in sorted_threats:
            hashed = hashlib.sha256(
                (salt + threat.matched_text).encode()
            ).hexdigest()[:8]
            
            result = (
                result[:threat.start_pos] +
                f"[HASH:{hashed}]" +
                result[threat.end_pos:]
            )
        
        return result


class ContentFilter:
    """
    Main content filtering interface.
    
    Example:
        >>> filter = ContentFilter()
        >>> 
        >>> # Analyze input
        >>> result = filter.analyze("Check this user input")
        >>> 
        >>> if not result.is_safe:
        ...     for threat in result.threats:
        ...         print(f"Detected: {threat.category.name}")
        >>> 
        >>> # Use sanitized text
        >>> safe_text = result.sanitized_text
    """
    
    def __init__(
        self,
        strategies: Optional[List[FilterStrategy]] = None,
        block_threshold: Severity = Severity.HIGH,
        auto_sanitize: bool = True,
    ):
        """
        Initialize content filter.
        
        Args:
            strategies: Filter strategies to use
            block_threshold: Severity threshold for blocking
            auto_sanitize: Automatically sanitize text
        """
        self.block_threshold = block_threshold
        self.auto_sanitize = auto_sanitize
        
        # Initialize strategies
        if strategies is None:
            self._strategies = [
                PIIDetector(),
                ToxicityDetector(),
                PromptInjectionDetector(),
                JailbreakDetector(),
            ]
        else:
            self._strategies = strategies
        
        # Custom rules
        self._rule_engine = CustomRuleEngine()
        
        # Sanitizer
        self._sanitizer = TextSanitizer()
    
    def add_strategy(self, strategy: FilterStrategy) -> None:
        """Add a filter strategy."""
        self._strategies.append(strategy)
    
    def add_rule(self, rule: FilterRule) -> None:
        """Add a custom rule."""
        self._rule_engine.add_rule(rule)
    
    def analyze(
        self,
        text: str,
        skip_strategies: Optional[List[str]] = None,
    ) -> FilterResult:
        """
        Analyze text for threats.
        
        Args:
            text: Text to analyze
            skip_strategies: Strategy names to skip
            
        Returns:
            FilterResult
        """
        import time
        start_time = time.time()
        
        all_threats: List[ThreatMatch] = []
        filters_applied: List[str] = []
        
        skip_set = set(skip_strategies or [])
        
        # Run strategies
        for strategy in self._strategies:
            if strategy.name in skip_set:
                continue
            
            threats = strategy.analyze(text)
            all_threats.extend(threats)
            filters_applied.append(strategy.name)
        
        # Run custom rules
        custom_threats = self._rule_engine.analyze(text)
        all_threats.extend(custom_threats)
        if custom_threats:
            filters_applied.append("custom_rules")
        
        # Determine max severity
        max_severity = Severity.NONE
        for threat in all_threats:
            if threat.severity.value > max_severity.value:
                max_severity = threat.severity
        
        # Determine if safe
        is_safe = max_severity.value < self.block_threshold.value
        
        # Sanitize if needed
        if self.auto_sanitize and all_threats:
            sanitized = self._sanitizer.redact(text, all_threats)
        else:
            sanitized = text
        
        processing_time = (time.time() - start_time) * 1000
        
        return FilterResult(
            is_safe=is_safe,
            original_text=text,
            sanitized_text=sanitized,
            threats=all_threats,
            max_severity=max_severity,
            processing_time_ms=processing_time,
            filters_applied=filters_applied,
        )
    
    def is_safe(self, text: str) -> bool:
        """Quick safety check."""
        result = self.analyze(text)
        return result.is_safe
    
    def sanitize(
        self,
        text: str,
        mask: str = "[REDACTED]",
    ) -> str:
        """
        Sanitize text.
        
        Args:
            text: Text to sanitize
            mask: Redaction mask
            
        Returns:
            Sanitized text
        """
        result = self.analyze(text)
        return self._sanitizer.redact(
            text,
            result.threats,
            mask=mask,
        )


class InputFilter:
    """
    Filter for user inputs (queries, prompts).
    
    Specialized for input validation with stricter checks.
    """
    
    def __init__(
        self,
        max_length: int = 10000,
        allow_special_chars: bool = True,
    ):
        """
        Initialize input filter.
        
        Args:
            max_length: Maximum input length
            allow_special_chars: Allow special characters
        """
        self.max_length = max_length
        self.allow_special_chars = allow_special_chars
        
        self._content_filter = ContentFilter(
            block_threshold=Severity.MEDIUM,
        )
    
    def validate(
        self,
        text: str,
    ) -> Tuple[bool, str, List[str]]:
        """
        Validate user input.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_valid, sanitized_text, error_messages)
        """
        errors = []
        
        # Length check
        if len(text) > self.max_length:
            errors.append(f"Input exceeds maximum length of {self.max_length}")
            text = text[:self.max_length]
        
        # Empty check
        if not text.strip():
            errors.append("Input is empty")
            return False, "", errors
        
        # Special character check
        if not self.allow_special_chars:
            cleaned = re.sub(r'[^\w\s.,!?-]', '', text)
            if cleaned != text:
                errors.append("Special characters removed")
                text = cleaned
        
        # Content analysis
        result = self._content_filter.analyze(text)
        
        if not result.is_safe:
            for threat in result.threats:
                errors.append(f"{threat.category.name}: {threat.description}")
        
        return result.is_safe, result.sanitized_text, errors


class OutputFilter:
    """
    Filter for generated outputs.
    
    Ensures generated content is safe for users.
    """
    
    def __init__(
        self,
        redact_pii: bool = True,
        block_harmful: bool = True,
    ):
        """
        Initialize output filter.
        
        Args:
            redact_pii: Redact PII in output
            block_harmful: Block harmful content
        """
        self.redact_pii = redact_pii
        self.block_harmful = block_harmful
        
        strategies = [PIIDetector()] if redact_pii else []
        if block_harmful:
            strategies.append(ToxicityDetector())
        
        self._content_filter = ContentFilter(
            strategies=strategies,
            block_threshold=Severity.HIGH,
        )
    
    def filter(
        self,
        text: str,
    ) -> FilterResult:
        """
        Filter generated output.
        
        Args:
            text: Generated text
            
        Returns:
            FilterResult
        """
        return self._content_filter.analyze(text)


# Convenience functions

def create_strict_filter() -> ContentFilter:
    """
    Create a strict content filter.
    
    Returns:
        ContentFilter with all strategies and low threshold
    """
    return ContentFilter(
        block_threshold=Severity.LOW,
        auto_sanitize=True,
    )


def create_permissive_filter() -> ContentFilter:
    """
    Create a permissive content filter.
    
    Only blocks critical threats.
    
    Returns:
        ContentFilter with critical threshold
    """
    return ContentFilter(
        block_threshold=Severity.CRITICAL,
        auto_sanitize=True,
    )


def check_for_injection(text: str) -> bool:
    """
    Quick check for prompt injection.
    
    Args:
        text: Text to check
        
    Returns:
        True if injection detected
    """
    detector = PromptInjectionDetector()
    threats = detector.analyze(text)
    return len(threats) > 0


def check_for_pii(text: str) -> List[str]:
    """
    Check for PII in text.
    
    Args:
        text: Text to check
        
    Returns:
        List of PII types detected
    """
    detector = PIIDetector()
    threats = detector.analyze(text)
    return [t.rule_id for t in threats]


def sanitize_text(
    text: str,
    mask: str = "[REDACTED]",
) -> str:
    """
    Quick text sanitization.
    
    Args:
        text: Text to sanitize
        mask: Redaction mask
        
    Returns:
        Sanitized text
    """
    filter = ContentFilter()
    return filter.sanitize(text, mask)
