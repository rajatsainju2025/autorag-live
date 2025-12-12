"""
Query planning and decomposition for AutoRAG-Live.

Provides multi-step query planning for complex questions
that require breaking down into sub-queries.

Features:
- Query complexity analysis
- Sub-query generation
- Dependency tracking
- Parallel execution planning
- Result aggregation

Example usage:
    >>> planner = QueryPlanner()
    >>> plan = planner.plan("Compare Python and JavaScript for web development")
    >>> for step in plan.steps:
    ...     print(f"{step.step_id}: {step.query}")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"


class StepType(str, Enum):
    """Types of query plan steps."""
    
    RETRIEVE = "retrieve"
    FILTER = "filter"
    COMPARE = "compare"
    AGGREGATE = "aggregate"
    SYNTHESIZE = "synthesize"
    VERIFY = "verify"


class DependencyType(str, Enum):
    """Types of dependencies between steps."""
    
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class PlanStep:
    """Represents a step in the query plan."""
    
    step_id: str
    query: str
    step_type: StepType
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution info
    priority: int = 0
    estimated_tokens: int = 0
    timeout: Optional[float] = None
    
    # Results
    result: Optional[Any] = None
    executed: bool = False
    execution_time: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """Complete query execution plan."""
    
    original_query: str
    steps: List[PlanStep]
    complexity: QueryComplexity
    
    # Execution order
    execution_groups: List[List[str]] = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    estimated_total_tokens: int = 0
    
    @property
    def step_count(self) -> int:
        """Get number of steps."""
        return len(self.steps)
    
    def get_step(self, step_id: str) -> Optional[PlanStep]:
        """Get step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_ready_steps(self) -> List[PlanStep]:
        """Get steps ready for execution (dependencies satisfied)."""
        ready = []
        executed_ids = {s.step_id for s in self.steps if s.executed}
        
        for step in self.steps:
            if step.executed:
                continue
            
            deps_satisfied = all(
                dep in executed_ids for dep in step.depends_on
            )
            
            if deps_satisfied:
                ready.append(step)
        
        return ready
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps are executed."""
        return all(s.executed for s in self.steps)


class QueryAnalyzer:
    """Analyze query complexity and structure."""
    
    # Complexity indicators
    COMPARISON_PATTERNS = [
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bbetter\b.*\bworse\b",
        r"\bpros?\s+and\s+cons?\b",
    ]
    
    MULTI_HOP_PATTERNS = [
        r"\bwho\b.*\bwhat\b",
        r"\bwhat\b.*\bwhy\b",
        r"\bfirst\b.*\bthen\b",
        r"\bafter\b.*\bbefore\b",
        r"\bresult\b.*\bcause\b",
    ]
    
    LIST_PATTERNS = [
        r"\blist\b",
        r"\btop\s+\d+\b",
        r"\ball\s+(?:the\s+)?\w+\b",
        r"\bevery\b",
        r"\beach\b",
    ]
    
    TEMPORAL_PATTERNS = [
        r"\bwhen\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bduring\b",
        r"\bhistory\b",
        r"\btimeline\b",
    ]
    
    def __init__(self):
        """Initialize query analyzer."""
        self._comparison_re = [
            re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS
        ]
        self._multi_hop_re = [
            re.compile(p, re.IGNORECASE) for p in self.MULTI_HOP_PATTERNS
        ]
        self._list_re = [
            re.compile(p, re.IGNORECASE) for p in self.LIST_PATTERNS
        ]
        self._temporal_re = [
            re.compile(p, re.IGNORECASE) for p in self.TEMPORAL_PATTERNS
        ]
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analyze query structure and complexity.
        
        Args:
            query: Input query
            
        Returns:
            Analysis results
        """
        analysis = {
            "query": query,
            "word_count": len(query.split()),
            "has_comparison": any(p.search(query) for p in self._comparison_re),
            "has_multi_hop": any(p.search(query) for p in self._multi_hop_re),
            "has_listing": any(p.search(query) for p in self._list_re),
            "has_temporal": any(p.search(query) for p in self._temporal_re),
            "question_count": query.count("?"),
            "entities": self._extract_entities(query),
            "complexity": QueryComplexity.SIMPLE,
        }
        
        # Determine complexity
        complexity_score = 0
        
        if analysis["has_comparison"]:
            complexity_score += 2
        if analysis["has_multi_hop"]:
            complexity_score += 3
        if analysis["has_listing"]:
            complexity_score += 1
        if analysis["has_temporal"]:
            complexity_score += 1
        if len(analysis["entities"]) > 2:
            complexity_score += 1
        if analysis["word_count"] > 20:
            complexity_score += 1
        if analysis["question_count"] > 1:
            complexity_score += 2
        
        if complexity_score >= 4:
            analysis["complexity"] = QueryComplexity.MULTI_HOP
        elif complexity_score >= 2:
            analysis["complexity"] = QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            analysis["complexity"] = QueryComplexity.MODERATE
        
        return analysis
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query."""
        # Simple capitalized word extraction
        words = query.split()
        entities = []
        
        for word in words:
            # Check for capitalized words (not at sentence start)
            clean = re.sub(r'[^\w]', '', word)
            if clean and clean[0].isupper() and len(clean) > 1:
                entities.append(clean)
        
        # Also extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        
        return list(set(entities))


class QueryDecomposer:
    """Decompose complex queries into sub-queries."""
    
    def __init__(self):
        """Initialize decomposer."""
        self.analyzer = QueryAnalyzer()
    
    def decompose(self, query: str) -> List[Tuple[str, StepType]]:
        """
        Decompose query into sub-queries.
        
        Args:
            query: Complex query
            
        Returns:
            List of (sub_query, step_type) tuples
        """
        analysis = self.analyzer.analyze(query)
        sub_queries = []
        
        # Handle comparison queries
        if analysis["has_comparison"]:
            sub_queries.extend(self._decompose_comparison(query, analysis))
        
        # Handle multi-hop queries
        elif analysis["has_multi_hop"]:
            sub_queries.extend(self._decompose_multi_hop(query, analysis))
        
        # Handle listing queries
        elif analysis["has_listing"]:
            sub_queries.extend(self._decompose_listing(query, analysis))
        
        # Simple query - no decomposition
        else:
            sub_queries.append((query, StepType.RETRIEVE))
        
        return sub_queries
    
    def _decompose_comparison(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Tuple[str, StepType]]:
        """Decompose comparison query."""
        sub_queries = []
        entities = analysis["entities"]
        
        if len(entities) >= 2:
            # Retrieve info about each entity
            for entity in entities[:2]:
                sub_queries.append(
                    (f"What are the key features of {entity}?", StepType.RETRIEVE)
                )
            
            # Add comparison step
            sub_queries.append(
                (f"Compare {' and '.join(entities[:2])}", StepType.COMPARE)
            )
        else:
            # Can't identify entities, use original
            sub_queries.append((query, StepType.RETRIEVE))
        
        return sub_queries
    
    def _decompose_multi_hop(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Tuple[str, StepType]]:
        """Decompose multi-hop query."""
        sub_queries = []
        
        # Split on common conjunctions
        parts = re.split(
            r'\s+(?:and\s+)?(?:then|after|before|therefore|because|since)\s+',
            query,
            flags=re.IGNORECASE
        )
        
        if len(parts) > 1:
            for part in parts:
                part = part.strip()
                if part:
                    sub_queries.append((part, StepType.RETRIEVE))
        else:
            # Try splitting on question words
            questions = re.split(r'\?\s*', query)
            for q in questions:
                q = q.strip()
                if q:
                    if not q.endswith("?"):
                        q += "?"
                    sub_queries.append((q, StepType.RETRIEVE))
        
        if len(sub_queries) > 1:
            # Add synthesis step
            sub_queries.append(
                ("Synthesize the findings", StepType.SYNTHESIZE)
            )
        elif not sub_queries:
            sub_queries.append((query, StepType.RETRIEVE))
        
        return sub_queries
    
    def _decompose_listing(
        self, query: str, analysis: Dict[str, Any]
    ) -> List[Tuple[str, StepType]]:
        """Decompose listing query."""
        sub_queries = []
        
        # Main retrieval
        sub_queries.append((query, StepType.RETRIEVE))
        
        # Add aggregation if asking for top N
        if re.search(r'\btop\s+\d+\b', query, re.IGNORECASE):
            sub_queries.append(
                ("Rank and filter results", StepType.FILTER)
            )
        
        # Add aggregation
        sub_queries.append(
            ("Aggregate and format results", StepType.AGGREGATE)
        )
        
        return sub_queries


class PlanOptimizer:
    """Optimize query execution plans."""
    
    def optimize(self, plan: QueryPlan) -> QueryPlan:
        """
        Optimize the execution plan.
        
        Args:
            plan: Query plan to optimize
            
        Returns:
            Optimized plan
        """
        # Identify parallelizable steps
        plan = self._identify_parallel_steps(plan)
        
        # Compute execution groups
        plan = self._compute_execution_groups(plan)
        
        # Estimate tokens
        plan = self._estimate_tokens(plan)
        
        return plan
    
    def _identify_parallel_steps(self, plan: QueryPlan) -> QueryPlan:
        """Identify steps that can run in parallel."""
        # Steps with same dependencies can be parallel
        dep_groups: Dict[tuple, List[str]] = {}
        
        for step in plan.steps:
            dep_key = tuple(sorted(step.depends_on))
            if dep_key not in dep_groups:
                dep_groups[dep_key] = []
            dep_groups[dep_key].append(step.step_id)
        
        return plan
    
    def _compute_execution_groups(self, plan: QueryPlan) -> QueryPlan:
        """Compute groups of steps that can execute together."""
        executed: Set[str] = set()
        groups: List[List[str]] = []
        
        while len(executed) < len(plan.steps):
            group = []
            
            for step in plan.steps:
                if step.step_id in executed:
                    continue
                
                # Check if dependencies are satisfied
                deps_satisfied = all(
                    dep in executed for dep in step.depends_on
                )
                
                if deps_satisfied:
                    group.append(step.step_id)
            
            if group:
                groups.append(group)
                executed.update(group)
            else:
                # Circular dependency or error
                remaining = [s.step_id for s in plan.steps if s.step_id not in executed]
                groups.append(remaining)
                break
        
        plan.execution_groups = groups
        return plan
    
    def _estimate_tokens(self, plan: QueryPlan) -> QueryPlan:
        """Estimate token usage for the plan."""
        total = 0
        
        for step in plan.steps:
            # Rough estimation based on query length
            estimated = len(step.query.split()) * 2 + 100  # Base overhead
            
            if step.step_type == StepType.SYNTHESIZE:
                estimated *= 2
            elif step.step_type == StepType.COMPARE:
                estimated *= 1.5
            
            step.estimated_tokens = int(estimated)
            total += step.estimated_tokens
        
        plan.estimated_total_tokens = total
        return plan


class QueryPlanner:
    """
    Main query planning interface.
    
    Example:
        >>> planner = QueryPlanner()
        >>> 
        >>> # Plan a complex query
        >>> plan = planner.plan("Compare Python and JavaScript for web development")
        >>> 
        >>> # Execute the plan
        >>> for group in plan.execution_groups:
        ...     for step_id in group:
        ...         step = plan.get_step(step_id)
        ...         result = execute_step(step)
        ...         step.result = result
        ...         step.executed = True
    """
    
    def __init__(
        self,
        max_steps: int = 10,
        enable_optimization: bool = True,
    ):
        """
        Initialize query planner.
        
        Args:
            max_steps: Maximum steps per plan
            enable_optimization: Enable plan optimization
        """
        self.max_steps = max_steps
        self.enable_optimization = enable_optimization
        
        self.analyzer = QueryAnalyzer()
        self.decomposer = QueryDecomposer()
        self.optimizer = PlanOptimizer()
    
    def plan(self, query: str) -> QueryPlan:
        """
        Create execution plan for a query.
        
        Args:
            query: User query
            
        Returns:
            QueryPlan
        """
        # Analyze query
        analysis = self.analyzer.analyze(query)
        
        # Decompose into sub-queries
        sub_queries = self.decomposer.decompose(query)
        
        # Limit steps
        sub_queries = sub_queries[:self.max_steps]
        
        # Create plan steps
        steps = []
        for i, (sub_query, step_type) in enumerate(sub_queries):
            step = PlanStep(
                step_id=f"step_{i}",
                query=sub_query,
                step_type=step_type,
                priority=len(sub_queries) - i,  # Earlier steps higher priority
            )
            
            # Add dependencies
            if i > 0 and step_type in (StepType.COMPARE, StepType.SYNTHESIZE, StepType.AGGREGATE):
                # These steps depend on previous retrieval steps
                step.depends_on = [
                    f"step_{j}" for j in range(i)
                    if sub_queries[j][1] == StepType.RETRIEVE
                ]
            
            steps.append(step)
        
        # Create plan
        plan = QueryPlan(
            original_query=query,
            steps=steps,
            complexity=analysis["complexity"],
        )
        
        # Optimize if enabled
        if self.enable_optimization:
            plan = self.optimizer.optimize(plan)
        
        return plan
    
    def is_decomposable(self, query: str) -> bool:
        """
        Check if query should be decomposed.
        
        Args:
            query: Query to check
            
        Returns:
            True if should decompose
        """
        analysis = self.analyzer.analyze(query)
        return analysis["complexity"] in (
            QueryComplexity.COMPLEX,
            QueryComplexity.MULTI_HOP,
        )
    
    def get_complexity(self, query: str) -> QueryComplexity:
        """
        Get query complexity.
        
        Args:
            query: Query to analyze
            
        Returns:
            QueryComplexity
        """
        analysis = self.analyzer.analyze(query)
        return analysis["complexity"]


class PlanExecutor:
    """Execute query plans."""
    
    def __init__(
        self,
        retrieval_func: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialize plan executor.
        
        Args:
            retrieval_func: Function to execute retrieval steps
        """
        self.retrieval_func = retrieval_func or self._default_retrieval
    
    async def execute(
        self,
        plan: QueryPlan,
        parallel: bool = True,
    ) -> QueryPlan:
        """
        Execute a query plan.
        
        Args:
            plan: Plan to execute
            parallel: Execute parallel steps concurrently
            
        Returns:
            Executed plan with results
        """
        import asyncio
        
        for group in plan.execution_groups:
            if parallel and len(group) > 1:
                # Execute group in parallel
                tasks = []
                for step_id in group:
                    step = plan.get_step(step_id)
                    if step:
                        tasks.append(self._execute_step(step, plan))
                
                await asyncio.gather(*tasks)
            else:
                # Execute sequentially
                for step_id in group:
                    step = plan.get_step(step_id)
                    if step:
                        await self._execute_step(step, plan)
        
        return plan
    
    async def _execute_step(
        self,
        step: PlanStep,
        plan: QueryPlan,
    ) -> None:
        """Execute a single step."""
        start_time = time.time()
        
        try:
            # Get dependency results
            dep_results = []
            for dep_id in step.depends_on:
                dep_step = plan.get_step(dep_id)
                if dep_step and dep_step.result:
                    dep_results.append(dep_step.result)
            
            # Execute based on step type
            if step.step_type == StepType.RETRIEVE:
                step.result = await self._execute_retrieval(step.query)
            
            elif step.step_type == StepType.COMPARE:
                step.result = await self._execute_comparison(step.query, dep_results)
            
            elif step.step_type == StepType.SYNTHESIZE:
                step.result = await self._execute_synthesis(step.query, dep_results)
            
            elif step.step_type == StepType.AGGREGATE:
                step.result = await self._execute_aggregation(step.query, dep_results)
            
            elif step.step_type == StepType.FILTER:
                step.result = await self._execute_filter(step.query, dep_results)
            
            else:
                step.result = await self._execute_retrieval(step.query)
            
            step.executed = True
            
        except Exception as e:
            logger.error(f"Step {step.step_id} failed: {e}")
            step.result = {"error": str(e)}
            step.executed = True
        
        step.execution_time = time.time() - start_time
    
    async def _execute_retrieval(self, query: str) -> Any:
        """Execute retrieval step."""
        if callable(self.retrieval_func):
            return self.retrieval_func(query)
        return {"query": query, "results": []}
    
    async def _execute_comparison(
        self, query: str, dep_results: List[Any]
    ) -> Any:
        """Execute comparison step."""
        return {
            "query": query,
            "type": "comparison",
            "inputs": dep_results,
        }
    
    async def _execute_synthesis(
        self, query: str, dep_results: List[Any]
    ) -> Any:
        """Execute synthesis step."""
        return {
            "query": query,
            "type": "synthesis",
            "inputs": dep_results,
        }
    
    async def _execute_aggregation(
        self, query: str, dep_results: List[Any]
    ) -> Any:
        """Execute aggregation step."""
        return {
            "query": query,
            "type": "aggregation",
            "inputs": dep_results,
        }
    
    async def _execute_filter(
        self, query: str, dep_results: List[Any]
    ) -> Any:
        """Execute filter step."""
        return {
            "query": query,
            "type": "filter",
            "inputs": dep_results,
        }
    
    def _default_retrieval(self, query: str) -> Any:
        """Default retrieval function."""
        return {"query": query, "results": []}


# Global planner instance
_default_planner: Optional[QueryPlanner] = None


def get_query_planner(
    max_steps: int = 10,
) -> QueryPlanner:
    """Get or create the default query planner."""
    global _default_planner
    if _default_planner is None:
        _default_planner = QueryPlanner(max_steps=max_steps)
    return _default_planner


def plan_query(query: str) -> QueryPlan:
    """
    Convenience function to plan a query.
    
    Args:
        query: User query
        
    Returns:
        QueryPlan
    """
    return get_query_planner().plan(query)


def analyze_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to analyze a query.
    
    Args:
        query: User query
        
    Returns:
        Analysis results
    """
    planner = get_query_planner()
    return planner.analyzer.analyze(query)
