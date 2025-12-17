"""
Pipeline builder module for AutoRAG-Live.

Provides a fluent API for building and configuring RAG pipelines
with chainable components and configuration management.

Features:
- Fluent builder pattern for pipeline construction
- Component chaining
- Configuration management
- Pipeline validation
- Execution orchestration
- Async pipeline support

Example usage:
    >>> pipeline = (
    ...     PipelineBuilder()
    ...     .with_retriever("dense", model="all-MiniLM-L6-v2")
    ...     .with_reranker("cross-encoder")
    ...     .with_generator("gpt-4")
    ...     .with_cache(enabled=True)
    ...     .build()
    ... )
    >>> result = pipeline.run("What is RAG?")
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ComponentType(Enum):
    """Types of pipeline components."""
    
    RETRIEVER = auto()
    RERANKER = auto()
    GENERATOR = auto()
    AUGMENTOR = auto()
    PREPROCESSOR = auto()
    POSTPROCESSOR = auto()
    CACHE = auto()
    VALIDATOR = auto()
    EVALUATOR = auto()
    CUSTOM = auto()


@dataclass
class ComponentConfig:
    """Configuration for a pipeline component."""
    
    component_type: ComponentType
    name: str
    
    # Component settings
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Execution settings
    timeout: Optional[float] = None
    retry_count: int = 0
    async_enabled: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline."""
    
    name: str = "rag_pipeline"
    
    # Components
    components: List[ComponentConfig] = field(default_factory=list)
    
    # Global settings
    cache_enabled: bool = True
    async_enabled: bool = True
    max_parallel: int = 4
    
    # Timeouts
    global_timeout: Optional[float] = None
    
    # Logging
    log_level: str = "INFO"
    trace_enabled: bool = False
    
    # Error handling
    fail_fast: bool = False
    fallback_enabled: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of a pipeline step."""
    
    step_name: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    
    # Timing
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    
    success: bool
    output: Any = None
    
    # Step results
    step_results: List[StepResult] = field(default_factory=list)
    
    # Timing
    total_time_ms: float = 0.0
    
    # Metadata
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get result for a specific step."""
        for step in self.step_results:
            if step.step_name == step_name:
                return step
        return None


@runtime_checkable
class PipelineComponent(Protocol):
    """Protocol for pipeline components."""
    
    @property
    def name(self) -> str:
        """Component name."""
        ...
    
    def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        ...


class BaseComponent(ABC):
    """Base class for pipeline components."""
    
    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize component.
        
        Args:
            name: Component name
            config: Component configuration
        """
        self._name = name
        self.config = config or {}
    
    @property
    def name(self) -> str:
        return self._name
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        pass
    
    async def aprocess(self, input_data: Any) -> Any:
        """Async process (default: wrap sync)."""
        return self.process(input_data)


class RetrieverComponent(BaseComponent):
    """Retriever component wrapper."""
    
    def __init__(
        self,
        name: str = "retriever",
        retriever_type: str = "dense",
        model: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(name, kwargs)
        self.retriever_type = retriever_type
        self.model = model
        self.top_k = top_k
    
    def process(self, input_data: Any) -> Any:
        """Retrieve documents."""
        # Placeholder - actual implementation would call retriever
        query = input_data.get('query', input_data) if isinstance(input_data, dict) else str(input_data)
        
        return {
            'query': query,
            'documents': [],
            'scores': [],
            'retriever': self.retriever_type,
        }


class RerankerComponent(BaseComponent):
    """Reranker component wrapper."""
    
    def __init__(
        self,
        name: str = "reranker",
        model: str = "cross-encoder",
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(name, kwargs)
        self.model = model
        self.top_k = top_k
    
    def process(self, input_data: Any) -> Any:
        """Rerank documents."""
        if not isinstance(input_data, dict):
            return input_data
        
        documents = input_data.get('documents', [])
        
        return {
            **input_data,
            'reranked_documents': documents[:self.top_k],
            'reranker': self.model,
        }


class GeneratorComponent(BaseComponent):
    """Generator component wrapper."""
    
    def __init__(
        self,
        name: str = "generator",
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__(name, kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def process(self, input_data: Any) -> Any:
        """Generate response."""
        if not isinstance(input_data, dict):
            return input_data
        
        query = input_data.get('query', '')
        documents = input_data.get('reranked_documents', input_data.get('documents', []))
        
        return {
            **input_data,
            'response': f"Generated response for: {query}",
            'generator': self.model,
            'context_used': len(documents),
        }


class PipelineStep:
    """A step in the pipeline."""
    
    def __init__(
        self,
        component: BaseComponent,
        config: ComponentConfig,
    ):
        """
        Initialize step.
        
        Args:
            component: Component to execute
            config: Step configuration
        """
        self.component = component
        self.config = config
    
    def execute(self, input_data: Any) -> StepResult:
        """Execute the step."""
        result = StepResult(
            step_name=self.config.name,
            success=False,
            start_time=time.time(),
        )
        
        if not self.config.enabled:
            result.success = True
            result.output = input_data
            result.end_time = time.time()
            result.metadata['skipped'] = True
            return result
        
        try:
            output = self.component.process(input_data)
            result.success = True
            result.output = output
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.output = input_data  # Pass through on error
            logger.error(f"Step {self.config.name} failed: {e}")
        
        result.end_time = time.time()
        return result
    
    async def aexecute(self, input_data: Any) -> StepResult:
        """Async execute the step."""
        result = StepResult(
            step_name=self.config.name,
            success=False,
            start_time=time.time(),
        )
        
        if not self.config.enabled:
            result.success = True
            result.output = input_data
            result.end_time = time.time()
            result.metadata['skipped'] = True
            return result
        
        try:
            if self.config.async_enabled:
                output = await self.component.aprocess(input_data)
            else:
                output = self.component.process(input_data)
            
            result.success = True
            result.output = output
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.output = input_data
            logger.error(f"Step {self.config.name} failed: {e}")
        
        result.end_time = time.time()
        return result


class Pipeline:
    """
    Executable RAG pipeline.
    
    Example:
        >>> # Build pipeline
        >>> pipeline = (
        ...     PipelineBuilder()
        ...     .with_retriever("dense")
        ...     .with_reranker("cross-encoder")
        ...     .with_generator("gpt-4")
        ...     .build()
        ... )
        >>> 
        >>> # Run pipeline
        >>> result = pipeline.run("What is machine learning?")
        >>> print(result.output['response'])
    """
    
    def __init__(
        self,
        config: PipelineConfig,
        steps: List[PipelineStep],
    ):
        """
        Initialize pipeline.
        
        Args:
            config: Pipeline configuration
            steps: Pipeline steps
        """
        self.config = config
        self.steps = steps
        self._is_running = False
    
    def run(
        self,
        query: Union[str, Dict[str, Any]],
        **kwargs,
    ) -> PipelineResult:
        """
        Run the pipeline synchronously.
        
        Args:
            query: Input query or data
            **kwargs: Additional parameters
            
        Returns:
            PipelineResult
        """
        start_time = time.time()
        
        result = PipelineResult(
            success=False,
            query=str(query),
        )
        
        # Prepare input
        if isinstance(query, str):
            current_data = {'query': query, **kwargs}
        else:
            current_data = {**query, **kwargs}
        
        # Execute steps
        self._is_running = True
        
        try:
            for step in self.steps:
                step_result = step.execute(current_data)
                result.step_results.append(step_result)
                
                if not step_result.success and self.config.fail_fast:
                    result.success = False
                    result.output = current_data
                    return result
                
                current_data = step_result.output
            
            result.success = True
            result.output = current_data
            
        finally:
            self._is_running = False
            result.total_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    async def arun(
        self,
        query: Union[str, Dict[str, Any]],
        **kwargs,
    ) -> PipelineResult:
        """
        Run the pipeline asynchronously.
        
        Args:
            query: Input query or data
            **kwargs: Additional parameters
            
        Returns:
            PipelineResult
        """
        start_time = time.time()
        
        result = PipelineResult(
            success=False,
            query=str(query),
        )
        
        # Prepare input
        if isinstance(query, str):
            current_data = {'query': query, **kwargs}
        else:
            current_data = {**query, **kwargs}
        
        # Execute steps
        self._is_running = True
        
        try:
            for step in self.steps:
                step_result = await step.aexecute(current_data)
                result.step_results.append(step_result)
                
                if not step_result.success and self.config.fail_fast:
                    result.success = False
                    result.output = current_data
                    return result
                
                current_data = step_result.output
            
            result.success = True
            result.output = current_data
            
        finally:
            self._is_running = False
            result.total_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def get_step(self, name: str) -> Optional[PipelineStep]:
        """Get step by name."""
        for step in self.steps:
            if step.config.name == name:
                return step
        return None
    
    @property
    def step_names(self) -> List[str]:
        """Get all step names."""
        return [s.config.name for s in self.steps]


class PipelineBuilder:
    """
    Fluent builder for RAG pipelines.
    
    Example:
        >>> # Simple pipeline
        >>> pipeline = (
        ...     PipelineBuilder("my_pipeline")
        ...     .with_retriever("dense", model="all-MiniLM-L6-v2", top_k=10)
        ...     .with_reranker("cross-encoder", top_k=5)
        ...     .with_generator("gpt-4", temperature=0.7)
        ...     .build()
        ... )
        >>> 
        >>> # Pipeline with cache and validation
        >>> pipeline = (
        ...     PipelineBuilder()
        ...     .with_cache(enabled=True, ttl=3600)
        ...     .with_retriever("hybrid")
        ...     .with_reranker("cohere")
        ...     .with_validator(check_pii=True)
        ...     .with_generator("claude-3")
        ...     .with_config(fail_fast=True, trace_enabled=True)
        ...     .build()
        ... )
    """
    
    def __init__(self, name: str = "rag_pipeline"):
        """
        Initialize builder.
        
        Args:
            name: Pipeline name
        """
        self._config = PipelineConfig(name=name)
        self._components: List[Tuple[ComponentConfig, BaseComponent]] = []
    
    def with_retriever(
        self,
        retriever_type: str = "dense",
        model: Optional[str] = None,
        top_k: int = 10,
        name: str = "retriever",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a retriever component.
        
        Args:
            retriever_type: Type of retriever
            model: Model name
            top_k: Number of documents to retrieve
            name: Component name
            **kwargs: Additional params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.RETRIEVER,
            name=name,
            params={
                'retriever_type': retriever_type,
                'model': model,
                'top_k': top_k,
                **kwargs,
            },
        )
        
        component = RetrieverComponent(
            name=name,
            retriever_type=retriever_type,
            model=model,
            top_k=top_k,
            **kwargs,
        )
        
        self._components.append((config, component))
        return self
    
    def with_reranker(
        self,
        model: str = "cross-encoder",
        top_k: int = 5,
        name: str = "reranker",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a reranker component.
        
        Args:
            model: Reranker model
            top_k: Number of documents to keep
            name: Component name
            **kwargs: Additional params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.RERANKER,
            name=name,
            params={
                'model': model,
                'top_k': top_k,
                **kwargs,
            },
        )
        
        component = RerankerComponent(
            name=name,
            model=model,
            top_k=top_k,
            **kwargs,
        )
        
        self._components.append((config, component))
        return self
    
    def with_generator(
        self,
        model: str = "gpt-4",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        name: str = "generator",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a generator component.
        
        Args:
            model: Generator model
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            name: Component name
            **kwargs: Additional params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.GENERATOR,
            name=name,
            params={
                'model': model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                **kwargs,
            },
        )
        
        component = GeneratorComponent(
            name=name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        
        self._components.append((config, component))
        return self
    
    def with_preprocessor(
        self,
        name: str = "preprocessor",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a preprocessor component.
        
        Args:
            name: Component name
            **kwargs: Preprocessor params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.PREPROCESSOR,
            name=name,
            params=kwargs,
        )
        
        # Placeholder component
        component = BaseComponentImpl(name, kwargs, lambda x: x)
        
        self._components.append((config, component))
        return self
    
    def with_postprocessor(
        self,
        name: str = "postprocessor",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a postprocessor component.
        
        Args:
            name: Component name
            **kwargs: Postprocessor params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.POSTPROCESSOR,
            name=name,
            params=kwargs,
        )
        
        # Placeholder component
        component = BaseComponentImpl(name, kwargs, lambda x: x)
        
        self._components.append((config, component))
        return self
    
    def with_cache(
        self,
        enabled: bool = True,
        ttl: int = 3600,
        name: str = "cache",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add cache configuration.
        
        Args:
            enabled: Enable caching
            ttl: Cache TTL in seconds
            name: Component name
            **kwargs: Cache params
            
        Returns:
            Self for chaining
        """
        self._config.cache_enabled = enabled
        
        config = ComponentConfig(
            component_type=ComponentType.CACHE,
            name=name,
            enabled=enabled,
            params={'ttl': ttl, **kwargs},
        )
        
        # Cache component (pass-through for now)
        component = BaseComponentImpl(name, {'ttl': ttl}, lambda x: x)
        
        self._components.append((config, component))
        return self
    
    def with_validator(
        self,
        check_pii: bool = True,
        check_toxicity: bool = True,
        name: str = "validator",
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a validator component.
        
        Args:
            check_pii: Check for PII
            check_toxicity: Check for toxicity
            name: Component name
            **kwargs: Validator params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.VALIDATOR,
            name=name,
            params={
                'check_pii': check_pii,
                'check_toxicity': check_toxicity,
                **kwargs,
            },
        )
        
        # Validator component
        component = BaseComponentImpl(
            name,
            {'check_pii': check_pii, 'check_toxicity': check_toxicity},
            lambda x: x,
        )
        
        self._components.append((config, component))
        return self
    
    def with_custom_component(
        self,
        component: BaseComponent,
        name: Optional[str] = None,
        **kwargs,
    ) -> PipelineBuilder:
        """
        Add a custom component.
        
        Args:
            component: Custom component instance
            name: Component name override
            **kwargs: Additional params
            
        Returns:
            Self for chaining
        """
        config = ComponentConfig(
            component_type=ComponentType.CUSTOM,
            name=name or component.name,
            params=kwargs,
        )
        
        self._components.append((config, component))
        return self
    
    def with_config(
        self,
        fail_fast: bool = False,
        async_enabled: bool = True,
        trace_enabled: bool = False,
        max_parallel: int = 4,
        global_timeout: Optional[float] = None,
        **kwargs,
    ) -> PipelineBuilder:
        """
        Configure pipeline settings.
        
        Args:
            fail_fast: Stop on first error
            async_enabled: Enable async execution
            trace_enabled: Enable tracing
            max_parallel: Max parallel operations
            global_timeout: Global timeout
            **kwargs: Additional settings
            
        Returns:
            Self for chaining
        """
        self._config.fail_fast = fail_fast
        self._config.async_enabled = async_enabled
        self._config.trace_enabled = trace_enabled
        self._config.max_parallel = max_parallel
        self._config.global_timeout = global_timeout
        self._config.metadata.update(kwargs)
        
        return self
    
    def enable_step(self, step_name: str) -> PipelineBuilder:
        """
        Enable a step by name.
        
        Args:
            step_name: Step to enable
            
        Returns:
            Self for chaining
        """
        for config, _ in self._components:
            if config.name == step_name:
                config.enabled = True
                break
        return self
    
    def disable_step(self, step_name: str) -> PipelineBuilder:
        """
        Disable a step by name.
        
        Args:
            step_name: Step to disable
            
        Returns:
            Self for chaining
        """
        for config, _ in self._components:
            if config.name == step_name:
                config.enabled = False
                break
        return self
    
    def validate(self) -> List[str]:
        """
        Validate pipeline configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for required components
        component_types = [c[0].component_type for c in self._components]
        
        if ComponentType.RETRIEVER not in component_types:
            errors.append("Pipeline requires at least one retriever")
        
        if ComponentType.GENERATOR not in component_types:
            errors.append("Pipeline requires at least one generator")
        
        # Check for duplicate names
        names = [c[0].name for c in self._components]
        if len(names) != len(set(names)):
            errors.append("Duplicate component names found")
        
        return errors
    
    def build(self, validate: bool = True) -> Pipeline:
        """
        Build the pipeline.
        
        Args:
            validate: Validate before building
            
        Returns:
            Built Pipeline
            
        Raises:
            ValueError: If validation fails
        """
        if validate:
            errors = self.validate()
            if errors:
                raise ValueError(f"Pipeline validation failed: {errors}")
        
        # Store component configs
        self._config.components = [c[0] for c in self._components]
        
        # Build steps
        steps = [
            PipelineStep(component, config)
            for config, component in self._components
        ]
        
        return Pipeline(self._config, steps)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> PipelineBuilder:
        """
        Create builder from config dict.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            PipelineBuilder
        """
        builder = cls(config.get('name', 'rag_pipeline'))
        
        for comp in config.get('components', []):
            comp_type = comp.get('type', '')
            params = comp.get('params', {})
            
            if comp_type == 'retriever':
                builder.with_retriever(**params)
            elif comp_type == 'reranker':
                builder.with_reranker(**params)
            elif comp_type == 'generator':
                builder.with_generator(**params)
            elif comp_type == 'cache':
                builder.with_cache(**params)
            elif comp_type == 'validator':
                builder.with_validator(**params)
        
        # Apply global config
        if 'settings' in config:
            builder.with_config(**config['settings'])
        
        return builder


class BaseComponentImpl(BaseComponent):
    """Simple component implementation for placeholders."""
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        process_fn: Callable[[Any], Any],
    ):
        super().__init__(name, config)
        self._process_fn = process_fn
    
    def process(self, input_data: Any) -> Any:
        return self._process_fn(input_data)


# Convenience functions

def create_simple_pipeline(
    retriever_model: str = "all-MiniLM-L6-v2",
    generator_model: str = "gpt-4",
) -> Pipeline:
    """
    Create a simple RAG pipeline.
    
    Args:
        retriever_model: Retriever model
        generator_model: Generator model
        
    Returns:
        Pipeline
    """
    return (
        PipelineBuilder()
        .with_retriever("dense", model=retriever_model)
        .with_generator(generator_model)
        .build()
    )


def create_full_pipeline(
    retriever_model: str = "all-MiniLM-L6-v2",
    reranker_model: str = "cross-encoder",
    generator_model: str = "gpt-4",
) -> Pipeline:
    """
    Create a full RAG pipeline with reranking.
    
    Args:
        retriever_model: Retriever model
        reranker_model: Reranker model
        generator_model: Generator model
        
    Returns:
        Pipeline
    """
    return (
        PipelineBuilder()
        .with_retriever("dense", model=retriever_model, top_k=20)
        .with_reranker(reranker_model, top_k=5)
        .with_generator(generator_model)
        .with_config(trace_enabled=True)
        .build()
    )


def create_production_pipeline(
    retriever_model: str = "all-MiniLM-L6-v2",
    reranker_model: str = "cross-encoder",
    generator_model: str = "gpt-4",
) -> Pipeline:
    """
    Create a production-ready RAG pipeline.
    
    Args:
        retriever_model: Retriever model
        reranker_model: Reranker model
        generator_model: Generator model
        
    Returns:
        Pipeline
    """
    return (
        PipelineBuilder("production_pipeline")
        .with_cache(enabled=True, ttl=3600)
        .with_retriever("hybrid", model=retriever_model, top_k=50)
        .with_reranker(reranker_model, top_k=10)
        .with_validator(check_pii=True, check_toxicity=True)
        .with_generator(generator_model, temperature=0.3)
        .with_config(
            fail_fast=False,
            trace_enabled=True,
            global_timeout=30.0,
        )
        .build()
    )
