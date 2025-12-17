"""
Response validation and safety module for AutoRAG-Live.

Provides comprehensive validation of LLM responses including
content safety, format validation, and output sanitization.

Features:
- Content safety checks
- Toxicity detection
- PII detection and masking
- Format validation (JSON, code, etc.)
- Length validation
- Factual consistency checking
- Response sanitization

Example usage:
    >>> validator = ResponseValidator()
    >>> result = validator.validate(response, query=original_query)
    >>> if result.is_valid:
    ...     print(result.sanitized_response)
    ... else:
    ...     print(result.issues)
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class IssueType(Enum):
    """Types of validation issues."""
    
    # Safety issues
    TOXIC_CONTENT = auto()
    HARMFUL_CONTENT = auto()
    PII_DETECTED = auto()
    INJECTION_ATTEMPT = auto()
    
    # Format issues
    INVALID_FORMAT = auto()
    INVALID_JSON = auto()
    INVALID_CODE = auto()
    MALFORMED_RESPONSE = auto()
    
    # Length issues
    TOO_SHORT = auto()
    TOO_LONG = auto()
    EMPTY_RESPONSE = auto()
    
    # Quality issues
    LOW_QUALITY = auto()
    OFF_TOPIC = auto()
    INCONSISTENT = auto()
    REPETITIVE = auto()
    
    # Factual issues
    POTENTIAL_HALLUCINATION = auto()
    UNSUPPORTED_CLAIM = auto()
    CONTRADICTION = auto()


@dataclass
class ValidationIssue:
    """A validation issue found in a response."""
    
    issue_type: IssueType
    severity: ValidationSeverity
    message: str
    
    # Position info
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    
    # Original text that caused issue
    problematic_text: Optional[str] = None
    
    # Suggested fix
    suggestion: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of response validation."""
    
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    # Original and sanitized
    original_response: str = ""
    sanitized_response: str = ""
    
    # Scores
    safety_score: float = 1.0
    quality_score: float = 1.0
    
    # Validation time
    validation_time_ms: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_critical_issues(self) -> bool:
        """Check for critical issues."""
        return any(
            i.severity == ValidationSeverity.CRITICAL
            for i in self.issues
        )
    
    def has_errors(self) -> bool:
        """Check for error-level issues."""
        return any(
            i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)
            for i in self.issues
        )
    
    def get_issues_by_type(
        self,
        issue_type: IssueType,
    ) -> List[ValidationIssue]:
        """Get issues of specific type."""
        return [i for i in self.issues if i.issue_type == issue_type]


class BaseValidator(ABC):
    """Base class for validators."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name."""
        pass
    
    @abstractmethod
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate response and return issues."""
        pass


class ToxicityValidator(BaseValidator):
    """Validates for toxic content."""
    
    # Toxic patterns (simplified - real impl would use ML model)
    TOXIC_PATTERNS = [
        # Profanity patterns (redacted)
        r'\b(hate|kill|attack|destroy)\s+(you|them|everyone)\b',
        r'\b(stupid|idiot|dumb)\s+(person|people|user)\b',
        # Add more patterns as needed
    ]
    
    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        """
        Initialize toxicity validator.
        
        Args:
            custom_patterns: Additional patterns to check
            case_sensitive: Whether patterns are case-sensitive
        """
        self.patterns = self.TOXIC_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        self.flags = 0 if case_sensitive else re.IGNORECASE
        self._compiled = [
            re.compile(p, self.flags) for p in self.patterns
        ]
    
    @property
    def name(self) -> str:
        return "toxicity"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Check for toxic content."""
        issues = []
        
        for pattern in self._compiled:
            for match in pattern.finditer(response):
                issues.append(ValidationIssue(
                    issue_type=IssueType.TOXIC_CONTENT,
                    severity=ValidationSeverity.ERROR,
                    message="Potentially toxic content detected",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    problematic_text=match.group(),
                ))
        
        return issues


class PIIValidator(BaseValidator):
    """Validates for Personally Identifiable Information."""
    
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    def __init__(
        self,
        detect_types: Optional[Set[str]] = None,
        custom_patterns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize PII validator.
        
        Args:
            detect_types: Types to detect (default: all)
            custom_patterns: Additional patterns
        """
        self.patterns = {}
        
        if detect_types:
            self.patterns = {
                k: v for k, v in self.PII_PATTERNS.items()
                if k in detect_types
            }
        else:
            self.patterns = self.PII_PATTERNS.copy()
        
        if custom_patterns:
            self.patterns.update(custom_patterns)
        
        self._compiled = {
            k: re.compile(v) for k, v in self.patterns.items()
        }
    
    @property
    def name(self) -> str:
        return "pii"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Check for PII."""
        issues = []
        
        for pii_type, pattern in self._compiled.items():
            for match in pattern.finditer(response):
                issues.append(ValidationIssue(
                    issue_type=IssueType.PII_DETECTED,
                    severity=ValidationSeverity.WARNING,
                    message=f"Potential {pii_type} detected",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    problematic_text=match.group(),
                    metadata={'pii_type': pii_type},
                ))
        
        return issues


class InjectionValidator(BaseValidator):
    """Validates for prompt injection attempts."""
    
    INJECTION_PATTERNS = [
        # Instruction override attempts
        r'ignore\s+(previous|all|above)\s+(instructions?|prompts?)',
        r'disregard\s+(previous|all|above)',
        r'forget\s+(everything|all|previous)',
        r'new\s+instruction[s]?:',
        r'system\s*:\s*you\s+are',
        r'<\|?(system|user|assistant)\|?>',
        r'\[INST\]|\[/INST\]',
        # Data exfiltration attempts
        r'print\s+(your|the)\s+(prompt|instructions?|system)',
        r'reveal\s+(your|the)\s+(prompt|system)',
        r'what\s+(are|is)\s+your\s+(instructions?|prompt)',
    ]
    
    def __init__(
        self,
        custom_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize injection validator.
        
        Args:
            custom_patterns: Additional patterns
        """
        self.patterns = self.INJECTION_PATTERNS.copy()
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        self._compiled = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]
    
    @property
    def name(self) -> str:
        return "injection"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Check for injection attempts."""
        issues = []
        
        for pattern in self._compiled:
            for match in pattern.finditer(response):
                issues.append(ValidationIssue(
                    issue_type=IssueType.INJECTION_ATTEMPT,
                    severity=ValidationSeverity.CRITICAL,
                    message="Potential prompt injection detected",
                    start_pos=match.start(),
                    end_pos=match.end(),
                    problematic_text=match.group(),
                ))
        
        return issues


class FormatValidator(BaseValidator):
    """Validates response format."""
    
    def __init__(
        self,
        expected_format: Optional[str] = None,
        json_schema: Optional[Dict] = None,
    ):
        """
        Initialize format validator.
        
        Args:
            expected_format: Expected format (json, markdown, code)
            json_schema: JSON schema for validation
        """
        self.expected_format = expected_format
        self.json_schema = json_schema
    
    @property
    def name(self) -> str:
        return "format"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate format."""
        issues = []
        
        if not self.expected_format:
            return issues
        
        if self.expected_format == 'json':
            issues.extend(self._validate_json(response))
        elif self.expected_format == 'code':
            issues.extend(self._validate_code(response, context))
        
        return issues
    
    def _validate_json(self, response: str) -> List[ValidationIssue]:
        """Validate JSON format."""
        issues = []
        
        # Try to extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        json_str = json_match.group(1) if json_match else response
        
        try:
            parsed = json.loads(json_str)
            
            # Validate against schema if provided
            if self.json_schema:
                # Simplified schema validation
                self._validate_schema(parsed, self.json_schema, issues)
                
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                issue_type=IssueType.INVALID_JSON,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid JSON: {e.msg}",
                metadata={'line': e.lineno, 'col': e.colno},
            ))
        
        return issues
    
    def _validate_schema(
        self,
        data: Any,
        schema: Dict,
        issues: List[ValidationIssue],
    ) -> None:
        """Simplified JSON schema validation."""
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'object' and not isinstance(data, dict):
                issues.append(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected object, got {type(data).__name__}",
                ))
            elif expected_type == 'array' and not isinstance(data, list):
                issues.append(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity=ValidationSeverity.ERROR,
                    message=f"Expected array, got {type(data).__name__}",
                ))
        
        if 'required' in schema and isinstance(data, dict):
            for field in schema['required']:
                if field not in data:
                    issues.append(ValidationIssue(
                        issue_type=IssueType.INVALID_FORMAT,
                        severity=ValidationSeverity.ERROR,
                        message=f"Missing required field: {field}",
                    ))
    
    def _validate_code(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate code format."""
        issues = []
        
        # Check for code blocks
        code_match = re.search(r'```(\w+)?\s*(.*?)\s*```', response, re.DOTALL)
        
        if not code_match:
            # Check if response looks like code without blocks
            code_indicators = ['def ', 'class ', 'import ', 'function ', 'const ']
            if any(ind in response for ind in code_indicators):
                issues.append(ValidationIssue(
                    issue_type=IssueType.INVALID_FORMAT,
                    severity=ValidationSeverity.WARNING,
                    message="Code should be wrapped in code blocks",
                    suggestion="Wrap code in ```language ... ``` blocks",
                ))
        
        return issues


class LengthValidator(BaseValidator):
    """Validates response length."""
    
    def __init__(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        min_words: int = 0,
        max_words: int = 0,
    ):
        """
        Initialize length validator.
        
        Args:
            min_length: Minimum character length
            max_length: Maximum character length
            min_words: Minimum word count (0 to disable)
            max_words: Maximum word count (0 to disable)
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_words = min_words
        self.max_words = max_words
    
    @property
    def name(self) -> str:
        return "length"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate length."""
        issues = []
        
        # Character length
        length = len(response.strip())
        
        if length == 0:
            issues.append(ValidationIssue(
                issue_type=IssueType.EMPTY_RESPONSE,
                severity=ValidationSeverity.ERROR,
                message="Empty response",
            ))
            return issues
        
        if length < self.min_length:
            issues.append(ValidationIssue(
                issue_type=IssueType.TOO_SHORT,
                severity=ValidationSeverity.WARNING,
                message=f"Response too short ({length} < {self.min_length})",
            ))
        
        if length > self.max_length:
            issues.append(ValidationIssue(
                issue_type=IssueType.TOO_LONG,
                severity=ValidationSeverity.WARNING,
                message=f"Response too long ({length} > {self.max_length})",
            ))
        
        # Word count
        words = len(response.split())
        
        if self.min_words > 0 and words < self.min_words:
            issues.append(ValidationIssue(
                issue_type=IssueType.TOO_SHORT,
                severity=ValidationSeverity.WARNING,
                message=f"Too few words ({words} < {self.min_words})",
            ))
        
        if self.max_words > 0 and words > self.max_words:
            issues.append(ValidationIssue(
                issue_type=IssueType.TOO_LONG,
                severity=ValidationSeverity.WARNING,
                message=f"Too many words ({words} > {self.max_words})",
            ))
        
        return issues


class QualityValidator(BaseValidator):
    """Validates response quality."""
    
    def __init__(
        self,
        check_repetition: bool = True,
        repetition_threshold: float = 0.3,
    ):
        """
        Initialize quality validator.
        
        Args:
            check_repetition: Check for repetitive content
            repetition_threshold: Threshold for repetition detection
        """
        self.check_repetition = check_repetition
        self.repetition_threshold = repetition_threshold
    
    @property
    def name(self) -> str:
        return "quality"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate quality."""
        issues = []
        
        if self.check_repetition:
            issues.extend(self._check_repetition(response))
        
        return issues
    
    def _check_repetition(self, response: str) -> List[ValidationIssue]:
        """Check for repetitive content."""
        issues = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip().lower() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return issues
        
        # Check for duplicate sentences
        seen = set()
        duplicates = 0
        
        for sentence in sentences:
            if sentence in seen:
                duplicates += 1
            seen.add(sentence)
        
        repetition_ratio = duplicates / len(sentences)
        
        if repetition_ratio > self.repetition_threshold:
            issues.append(ValidationIssue(
                issue_type=IssueType.REPETITIVE,
                severity=ValidationSeverity.WARNING,
                message=f"High repetition detected ({repetition_ratio:.1%})",
                metadata={'repetition_ratio': repetition_ratio},
            ))
        
        # Check for repeated phrases (n-grams)
        words = response.lower().split()
        if len(words) > 10:
            trigrams = [
                ' '.join(words[i:i+3])
                for i in range(len(words) - 2)
            ]
            trigram_counts = {}
            for tg in trigrams:
                trigram_counts[tg] = trigram_counts.get(tg, 0) + 1
            
            repeated = [tg for tg, count in trigram_counts.items() if count > 3]
            if repeated:
                issues.append(ValidationIssue(
                    issue_type=IssueType.REPETITIVE,
                    severity=ValidationSeverity.INFO,
                    message=f"Repeated phrases detected: {len(repeated)}",
                    metadata={'repeated_phrases': repeated[:5]},
                ))
        
        return issues


class FactualValidator(BaseValidator):
    """Validates factual consistency with context."""
    
    def __init__(
        self,
        check_contradictions: bool = True,
        check_unsupported: bool = True,
    ):
        """
        Initialize factual validator.
        
        Args:
            check_contradictions: Check for contradictions
            check_unsupported: Check for unsupported claims
        """
        self.check_contradictions = check_contradictions
        self.check_unsupported = check_unsupported
    
    @property
    def name(self) -> str:
        return "factual"
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationIssue]:
        """Validate factual consistency."""
        issues = []
        
        if not context:
            return issues
        
        source_texts = context.get('source_texts', [])
        query = context.get('query', '')
        
        if not source_texts:
            return issues
        
        # Simple heuristic checks
        # Real implementation would use NLI models
        
        # Check for hallucination indicators
        hallucination_indicators = [
            r'I\s+(think|believe|imagine)',
            r'it\s+is\s+(likely|probably|possibly)',
            r'as\s+far\s+as\s+I\s+know',
            r'I\s+don\'t\s+have\s+information',
        ]
        
        for pattern in hallucination_indicators:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(ValidationIssue(
                    issue_type=IssueType.POTENTIAL_HALLUCINATION,
                    severity=ValidationSeverity.INFO,
                    message="Response contains uncertainty indicators",
                ))
                break
        
        return issues


class ResponseSanitizer:
    """Sanitizes responses by removing/masking issues."""
    
    def __init__(
        self,
        mask_pii: bool = True,
        remove_toxic: bool = True,
        pii_mask: str = "[REDACTED]",
    ):
        """
        Initialize sanitizer.
        
        Args:
            mask_pii: Mask PII
            remove_toxic: Remove toxic content
            pii_mask: String to use for PII masking
        """
        self.mask_pii = mask_pii
        self.remove_toxic = remove_toxic
        self.pii_mask = pii_mask
    
    def sanitize(
        self,
        response: str,
        issues: List[ValidationIssue],
    ) -> str:
        """
        Sanitize response based on issues.
        
        Args:
            response: Original response
            issues: Validation issues
            
        Returns:
            Sanitized response
        """
        sanitized = response
        
        # Sort issues by position (reverse) to handle from end to start
        positioned_issues = [
            i for i in issues
            if i.start_pos is not None and i.end_pos is not None
        ]
        positioned_issues.sort(key=lambda i: i.start_pos, reverse=True)
        
        for issue in positioned_issues:
            if self.mask_pii and issue.issue_type == IssueType.PII_DETECTED:
                sanitized = (
                    sanitized[:issue.start_pos] +
                    self.pii_mask +
                    sanitized[issue.end_pos:]
                )
            elif self.remove_toxic and issue.issue_type == IssueType.TOXIC_CONTENT:
                sanitized = (
                    sanitized[:issue.start_pos] +
                    sanitized[issue.end_pos:]
                )
        
        # Clean up extra whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized


class ResponseValidator:
    """
    Main response validation interface.
    
    Example:
        >>> validator = ResponseValidator()
        >>> 
        >>> result = validator.validate(
        ...     "Here is the answer with john@email.com contact",
        ...     context={'query': 'What is the process?'}
        ... )
        >>> 
        >>> if result.is_valid:
        ...     print("Response is valid")
        ... else:
        ...     for issue in result.issues:
        ...         print(f"Issue: {issue.message}")
        >>> 
        >>> # Get sanitized version
        >>> print(result.sanitized_response)
    """
    
    def __init__(
        self,
        validators: Optional[List[BaseValidator]] = None,
        sanitize: bool = True,
        fail_on_critical: bool = True,
        fail_on_error: bool = False,
    ):
        """
        Initialize response validator.
        
        Args:
            validators: Custom validators (default: all built-in)
            sanitize: Sanitize response
            fail_on_critical: Mark invalid if critical issues
            fail_on_error: Mark invalid if error-level issues
        """
        self.validators = validators or self._default_validators()
        self.sanitize = sanitize
        self.fail_on_critical = fail_on_critical
        self.fail_on_error = fail_on_error
        
        self._sanitizer = ResponseSanitizer()
    
    def _default_validators(self) -> List[BaseValidator]:
        """Get default validators."""
        return [
            ToxicityValidator(),
            PIIValidator(),
            InjectionValidator(),
            LengthValidator(),
            QualityValidator(),
            FactualValidator(),
        ]
    
    def validate(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a response.
        
        Args:
            response: Response to validate
            context: Validation context
            
        Returns:
            ValidationResult
        """
        import time
        start_time = time.time()
        
        result = ValidationResult(
            is_valid=True,
            original_response=response,
            sanitized_response=response,
        )
        
        # Run all validators
        for validator in self.validators:
            try:
                issues = validator.validate(response, context)
                result.issues.extend(issues)
            except Exception as e:
                logger.warning(
                    f"Validator {validator.name} failed: {e}"
                )
        
        # Determine validity
        if self.fail_on_critical and result.has_critical_issues():
            result.is_valid = False
        
        if self.fail_on_error and result.has_errors():
            result.is_valid = False
        
        # Calculate scores
        result.safety_score = self._calculate_safety_score(result.issues)
        result.quality_score = self._calculate_quality_score(result.issues)
        
        # Sanitize if requested
        if self.sanitize:
            result.sanitized_response = self._sanitizer.sanitize(
                response, result.issues
            )
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _calculate_safety_score(
        self,
        issues: List[ValidationIssue],
    ) -> float:
        """Calculate safety score (0-1)."""
        safety_issues = [
            i for i in issues
            if i.issue_type in (
                IssueType.TOXIC_CONTENT,
                IssueType.HARMFUL_CONTENT,
                IssueType.PII_DETECTED,
                IssueType.INJECTION_ATTEMPT,
            )
        ]
        
        if not safety_issues:
            return 1.0
        
        # Deduct based on severity
        penalty = 0.0
        for issue in safety_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                penalty += 0.5
            elif issue.severity == ValidationSeverity.ERROR:
                penalty += 0.25
            elif issue.severity == ValidationSeverity.WARNING:
                penalty += 0.1
        
        return max(0.0, 1.0 - penalty)
    
    def _calculate_quality_score(
        self,
        issues: List[ValidationIssue],
    ) -> float:
        """Calculate quality score (0-1)."""
        quality_issues = [
            i for i in issues
            if i.issue_type in (
                IssueType.LOW_QUALITY,
                IssueType.REPETITIVE,
                IssueType.TOO_SHORT,
                IssueType.TOO_LONG,
                IssueType.OFF_TOPIC,
            )
        ]
        
        if not quality_issues:
            return 1.0
        
        penalty = 0.0
        for issue in quality_issues:
            if issue.severity == ValidationSeverity.ERROR:
                penalty += 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                penalty += 0.15
            else:
                penalty += 0.05
        
        return max(0.0, 1.0 - penalty)
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator."""
        self.validators.append(validator)
    
    def remove_validator(self, name: str) -> bool:
        """Remove validator by name."""
        for i, v in enumerate(self.validators):
            if v.name == name:
                del self.validators[i]
                return True
        return False


# Convenience functions

def validate_response(
    response: str,
    context: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """
    Quick response validation.
    
    Args:
        response: Response to validate
        context: Optional context
        
    Returns:
        ValidationResult
    """
    validator = ResponseValidator()
    return validator.validate(response, context)


def sanitize_response(
    response: str,
    mask_pii: bool = True,
) -> str:
    """
    Quick response sanitization.
    
    Args:
        response: Response to sanitize
        mask_pii: Mask PII
        
    Returns:
        Sanitized response
    """
    validator = ResponseValidator(sanitize=True)
    result = validator.validate(response)
    return result.sanitized_response


def is_safe(response: str) -> bool:
    """
    Quick safety check.
    
    Args:
        response: Response to check
        
    Returns:
        True if safe
    """
    result = validate_response(response)
    return result.safety_score >= 0.8


def check_pii(response: str) -> List[Tuple[str, str, int, int]]:
    """
    Check for PII in response.
    
    Args:
        response: Response to check
        
    Returns:
        List of (pii_type, text, start, end)
    """
    validator = PIIValidator()
    issues = validator.validate(response)
    
    return [
        (
            issue.metadata.get('pii_type', 'unknown'),
            issue.problematic_text,
            issue.start_pos,
            issue.end_pos,
        )
        for issue in issues
        if issue.problematic_text
    ]
