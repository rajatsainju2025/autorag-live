"""
Evaluation reporting utilities for AutoRAG-Live.

Provides comprehensive evaluation report generation with
visualizations, comparisons, and export capabilities.

Features:
- Report generation in multiple formats
- Metric aggregation and statistics
- Comparison reports
- Trend analysis
- Export to HTML, JSON, Markdown
- Dashboard generation

Example usage:
    >>> reporter = EvaluationReporter()
    >>> report = reporter.generate_report(eval_results)
    >>> reporter.export_html(report, "report.html")
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Report output formats."""
    
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    CSV = "csv"


class MetricType(str, Enum):
    """Types of evaluation metrics."""
    
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    LATENCY = "latency"
    QUALITY = "quality"
    COST = "cost"


@dataclass
class MetricValue:
    """A single metric value."""
    
    name: str
    value: float
    
    # Classification
    type: MetricType = MetricType.QUALITY
    
    # Statistics
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    count: int = 1
    
    # Display
    unit: str = ""
    format_str: str = ".3f"
    higher_is_better: bool = True
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def formatted(self) -> str:
        """Get formatted value."""
        return f"{self.value:{self.format_str}}{self.unit}"
    
    def compare_to(self, other: MetricValue) -> float:
        """Compare to another metric value."""
        diff = self.value - other.value
        if not self.higher_is_better:
            diff = -diff
        return diff


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    
    name: str
    metrics: Dict[str, MetricValue]
    
    # Run info
    run_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Details
    sample_count: int = 0
    success_count: int = 0
    error_count: int = 0
    
    # Per-sample results
    sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.sample_count > 0:
            return self.success_count / self.sample_count
        return 0.0
    
    def get_metric(self, name: str) -> Optional[MetricValue]:
        """Get metric by name."""
        return self.metrics.get(name)


@dataclass
class ComparisonResult:
    """Result of comparing evaluation runs."""
    
    baseline: EvaluationResult
    comparison: EvaluationResult
    
    # Metric differences
    metric_diffs: Dict[str, float] = field(default_factory=dict)
    
    # Improvements
    improvements: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    
    # Statistical significance
    significant_changes: Dict[str, bool] = field(default_factory=dict)
    
    # Summary
    overall_improvement: float = 0.0


@dataclass
class ReportSection:
    """A section of the report."""
    
    title: str
    content: str
    
    # Subsections
    subsections: List[ReportSection] = field(default_factory=list)
    
    # Data
    tables: List[Dict[str, Any]] = field(default_factory=list)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    level: int = 1
    order: int = 0


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    
    title: str
    sections: List[ReportSection]
    
    # Results
    results: List[EvaluationResult] = field(default_factory=list)
    comparisons: List[ComparisonResult] = field(default_factory=list)
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    
    # Summary
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricAggregator:
    """Aggregate metrics across samples."""
    
    @staticmethod
    def aggregate(
        values: List[float],
        name: str,
        metric_type: MetricType = MetricType.QUALITY,
        **kwargs,
    ) -> MetricValue:
        """
        Aggregate a list of values.
        
        Args:
            values: List of metric values
            name: Metric name
            metric_type: Type of metric
            **kwargs: Additional MetricValue parameters
            
        Returns:
            Aggregated MetricValue
        """
        if not values:
            return MetricValue(name=name, value=0.0, type=metric_type, **kwargs)
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        
        return MetricValue(
            name=name,
            value=mean,
            type=metric_type,
            std=std,
            min_val=min(values),
            max_val=max(values),
            count=len(values),
            **kwargs,
        )
    
    @staticmethod
    def aggregate_results(
        results: List[EvaluationResult],
    ) -> Dict[str, MetricValue]:
        """Aggregate metrics across multiple results."""
        all_metrics: Dict[str, List[float]] = {}
        
        for result in results:
            for name, metric in result.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(metric.value)
        
        aggregated = {}
        for name, values in all_metrics.items():
            aggregated[name] = MetricAggregator.aggregate(values, name)
        
        return aggregated


class ReportFormatter(ABC):
    """Base class for report formatters."""
    
    @abstractmethod
    def format(self, report: EvaluationReport) -> str:
        """Format report to string."""
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class HTMLFormatter(ReportFormatter):
    """Format report as HTML."""
    
    def format(self, report: EvaluationReport) -> str:
        """Format as HTML."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{report.title}</title>",
            self._get_styles(),
            "</head>",
            "<body>",
            f'<div class="container">',
            f"<h1>{report.title}</h1>",
            f'<p class="timestamp">Generated: {report.generated_at.strftime("%Y-%m-%d %H:%M:%S")}</p>',
        ]
        
        # Summary
        if report.summary:
            html.append(f'<div class="summary"><h2>Summary</h2><p>{report.summary}</p></div>')
        
        # Sections
        for section in report.sections:
            html.append(self._format_section(section))
        
        # Results tables
        for result in report.results:
            html.append(self._format_result_table(result))
        
        # Recommendations
        if report.recommendations:
            html.append('<div class="recommendations"><h2>Recommendations</h2><ul>')
            for rec in report.recommendations:
                html.append(f"<li>{rec}</li>")
            html.append("</ul></div>")
        
        html.extend(["</div>", "</body>", "</html>"])
        
        return "\n".join(html)
    
    def _get_styles(self) -> str:
        """Get CSS styles."""
        return """
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
            h2 { color: #555; margin-top: 30px; }
            h3 { color: #666; }
            .timestamp { color: #888; font-size: 14px; }
            .summary { background: #e3f2fd; padding: 15px; border-radius: 4px; margin: 20px 0; }
            table { width: 100%; border-collapse: collapse; margin: 20px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background: #f8f9fa; font-weight: 600; }
            tr:hover { background: #f5f5f5; }
            .metric-good { color: #28a745; }
            .metric-bad { color: #dc3545; }
            .recommendations { background: #fff3cd; padding: 15px; border-radius: 4px; margin: 20px 0; }
            .recommendations ul { margin: 10px 0; padding-left: 20px; }
            .chart { margin: 20px 0; text-align: center; }
        </style>
        """
    
    def _format_section(self, section: ReportSection) -> str:
        """Format a section."""
        tag = f"h{min(section.level + 1, 6)}"
        html = [f"<{tag}>{section.title}</{tag}>"]
        
        if section.content:
            html.append(f"<p>{section.content}</p>")
        
        # Tables
        for table in section.tables:
            html.append(self._format_table(table))
        
        # Subsections
        for subsection in section.subsections:
            html.append(self._format_section(subsection))
        
        return "\n".join(html)
    
    def _format_table(self, table: Dict[str, Any]) -> str:
        """Format a table."""
        html = ['<table>']
        
        # Headers
        if 'headers' in table:
            html.append('<tr>')
            for header in table['headers']:
                html.append(f'<th>{header}</th>')
            html.append('</tr>')
        
        # Rows
        for row in table.get('rows', []):
            html.append('<tr>')
            for cell in row:
                html.append(f'<td>{cell}</td>')
            html.append('</tr>')
        
        html.append('</table>')
        return "\n".join(html)
    
    def _format_result_table(self, result: EvaluationResult) -> str:
        """Format evaluation result as table."""
        html = [
            f"<h3>Results: {result.name}</h3>",
            "<table>",
            "<tr><th>Metric</th><th>Value</th><th>Type</th></tr>",
        ]
        
        for name, metric in sorted(result.metrics.items()):
            value_class = "metric-good" if metric.value >= 0.7 else ""
            html.append(
                f'<tr><td>{name}</td>'
                f'<td class="{value_class}">{metric.formatted}</td>'
                f'<td>{metric.type.value}</td></tr>'
            )
        
        html.append("</table>")
        return "\n".join(html)


class MarkdownFormatter(ReportFormatter):
    """Format report as Markdown."""
    
    def format(self, report: EvaluationReport) -> str:
        """Format as Markdown."""
        lines = [
            f"# {report.title}",
            "",
            f"*Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]
        
        # Summary
        if report.summary:
            lines.extend([
                "## Summary",
                "",
                report.summary,
                "",
            ])
        
        # Sections
        for section in report.sections:
            lines.append(self._format_section(section))
        
        # Results
        for result in report.results:
            lines.append(self._format_result(result))
        
        # Recommendations
        if report.recommendations:
            lines.extend(["## Recommendations", ""])
            for rec in report.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _format_section(self, section: ReportSection) -> str:
        """Format section as Markdown."""
        prefix = "#" * (section.level + 1)
        lines = [f"{prefix} {section.title}", ""]
        
        if section.content:
            lines.extend([section.content, ""])
        
        # Tables
        for table in section.tables:
            lines.append(self._format_table(table))
        
        # Subsections
        for subsection in section.subsections:
            lines.append(self._format_section(subsection))
        
        return "\n".join(lines)
    
    def _format_table(self, table: Dict[str, Any]) -> str:
        """Format table as Markdown."""
        lines = []
        
        headers = table.get('headers', [])
        if headers:
            lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            lines.append("| " + " | ".join("---" for _ in headers) + " |")
        
        for row in table.get('rows', []):
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        lines.append("")
        return "\n".join(lines)
    
    def _format_result(self, result: EvaluationResult) -> str:
        """Format evaluation result."""
        lines = [
            f"### {result.name}",
            "",
            "| Metric | Value | Type |",
            "| --- | --- | --- |",
        ]
        
        for name, metric in sorted(result.metrics.items()):
            lines.append(f"| {name} | {metric.formatted} | {metric.type.value} |")
        
        lines.append("")
        return "\n".join(lines)


class JSONFormatter(ReportFormatter):
    """Format report as JSON."""
    
    def format(self, report: EvaluationReport) -> str:
        """Format as JSON."""
        data = {
            "title": report.title,
            "generated_at": report.generated_at.isoformat(),
            "version": report.version,
            "summary": report.summary,
            "results": [
                self._format_result(r) for r in report.results
            ],
            "comparisons": [
                self._format_comparison(c) for c in report.comparisons
            ],
            "recommendations": report.recommendations,
            "metadata": report.metadata,
        }
        
        return json.dumps(data, indent=2)
    
    def _format_result(self, result: EvaluationResult) -> Dict:
        """Format result as dict."""
        return {
            "name": result.name,
            "run_id": result.run_id,
            "timestamp": result.timestamp.isoformat(),
            "metrics": {
                name: {
                    "value": m.value,
                    "std": m.std,
                    "type": m.type.value,
                }
                for name, m in result.metrics.items()
            },
            "sample_count": result.sample_count,
            "success_rate": result.success_rate,
            "config": result.config,
        }
    
    def _format_comparison(self, comparison: ComparisonResult) -> Dict:
        """Format comparison as dict."""
        return {
            "baseline": comparison.baseline.name,
            "comparison": comparison.comparison.name,
            "metric_diffs": comparison.metric_diffs,
            "improvements": comparison.improvements,
            "regressions": comparison.regressions,
            "overall_improvement": comparison.overall_improvement,
        }


class TextFormatter(ReportFormatter):
    """Format report as plain text."""
    
    def format(self, report: EvaluationReport) -> str:
        """Format as text."""
        lines = [
            "=" * 60,
            report.title.center(60),
            "=" * 60,
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
        
        if report.summary:
            lines.extend(["SUMMARY", "-" * 40, report.summary, ""])
        
        for result in report.results:
            lines.extend([
                f"RESULTS: {result.name}",
                "-" * 40,
            ])
            for name, metric in sorted(result.metrics.items()):
                lines.append(f"  {name}: {metric.formatted}")
            lines.append("")
        
        if report.recommendations:
            lines.extend(["RECOMMENDATIONS", "-" * 40])
            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"  {i}. {rec}")
        
        return "\n".join(lines)


class ReportComparator:
    """Compare evaluation results."""
    
    def __init__(
        self,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize comparator.
        
        Args:
            significance_threshold: Threshold for significance
        """
        self.significance_threshold = significance_threshold
    
    def compare(
        self,
        baseline: EvaluationResult,
        comparison: EvaluationResult,
    ) -> ComparisonResult:
        """
        Compare two evaluation results.
        
        Args:
            baseline: Baseline result
            comparison: Comparison result
            
        Returns:
            ComparisonResult
        """
        metric_diffs = {}
        improvements = []
        regressions = []
        significant = {}
        
        # Compare each metric
        all_metrics = set(baseline.metrics.keys()) | set(comparison.metrics.keys())
        
        for name in all_metrics:
            base_metric = baseline.metrics.get(name)
            comp_metric = comparison.metrics.get(name)
            
            if base_metric and comp_metric:
                diff = comp_metric.compare_to(base_metric)
                metric_diffs[name] = diff
                
                # Check for improvement/regression
                if abs(diff) > self.significance_threshold:
                    significant[name] = True
                    if diff > 0:
                        improvements.append(name)
                    else:
                        regressions.append(name)
                else:
                    significant[name] = False
        
        # Overall improvement
        if metric_diffs:
            overall = sum(metric_diffs.values()) / len(metric_diffs)
        else:
            overall = 0.0
        
        return ComparisonResult(
            baseline=baseline,
            comparison=comparison,
            metric_diffs=metric_diffs,
            improvements=improvements,
            regressions=regressions,
            significant_changes=significant,
            overall_improvement=overall,
        )


class EvaluationReporter:
    """
    Main evaluation reporting interface.
    
    Example:
        >>> reporter = EvaluationReporter()
        >>> 
        >>> # Generate report
        >>> report = reporter.generate_report(
        ...     results=[eval_result],
        ...     title="RAG Evaluation Report"
        ... )
        >>> 
        >>> # Export to HTML
        >>> reporter.export(report, "report.html", format="html")
        >>> 
        >>> # Compare results
        >>> comparison = reporter.compare(baseline, new_result)
    """
    
    def __init__(
        self,
        default_format: ReportFormat = ReportFormat.HTML,
    ):
        """
        Initialize reporter.
        
        Args:
            default_format: Default output format
        """
        self.default_format = default_format
        
        # Formatters
        self.formatters = {
            ReportFormat.HTML: HTMLFormatter(),
            ReportFormat.MARKDOWN: MarkdownFormatter(),
            ReportFormat.JSON: JSONFormatter(),
            ReportFormat.TEXT: TextFormatter(),
        }
        
        # Comparator
        self.comparator = ReportComparator()
    
    def generate_report(
        self,
        results: List[EvaluationResult],
        title: str = "Evaluation Report",
        include_comparisons: bool = True,
        include_recommendations: bool = True,
    ) -> EvaluationReport:
        """
        Generate evaluation report.
        
        Args:
            results: List of evaluation results
            title: Report title
            include_comparisons: Include comparisons if multiple results
            include_recommendations: Include recommendations
            
        Returns:
            EvaluationReport
        """
        sections = []
        comparisons = []
        recommendations = []
        
        # Overview section
        sections.append(self._create_overview_section(results))
        
        # Metrics section
        sections.append(self._create_metrics_section(results))
        
        # Comparisons
        if include_comparisons and len(results) > 1:
            for i in range(len(results) - 1):
                comp = self.comparator.compare(results[i], results[i + 1])
                comparisons.append(comp)
            
            sections.append(self._create_comparison_section(comparisons))
        
        # Generate recommendations
        if include_recommendations:
            recommendations = self._generate_recommendations(results, comparisons)
        
        # Summary
        summary = self._generate_summary(results, comparisons)
        
        return EvaluationReport(
            title=title,
            sections=sections,
            results=results,
            comparisons=comparisons,
            summary=summary,
            recommendations=recommendations,
        )
    
    def export(
        self,
        report: EvaluationReport,
        path: Union[str, Path],
        format: Optional[Union[str, ReportFormat]] = None,
    ) -> None:
        """
        Export report to file.
        
        Args:
            report: Report to export
            path: Output path
            format: Output format
        """
        if format is None:
            format = self.default_format
        elif isinstance(format, str):
            format = ReportFormat(format)
        
        formatter = self.formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        
        content = formatter.format(report)
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        
        logger.info(f"Report exported to {path}")
    
    def compare(
        self,
        baseline: EvaluationResult,
        comparison: EvaluationResult,
    ) -> ComparisonResult:
        """
        Compare two evaluation results.
        
        Args:
            baseline: Baseline result
            comparison: Comparison result
            
        Returns:
            ComparisonResult
        """
        return self.comparator.compare(baseline, comparison)
    
    def format_report(
        self,
        report: EvaluationReport,
        format: Union[str, ReportFormat] = ReportFormat.HTML,
    ) -> str:
        """
        Format report as string.
        
        Args:
            report: Report to format
            format: Output format
            
        Returns:
            Formatted report string
        """
        if isinstance(format, str):
            format = ReportFormat(format)
        
        formatter = self.formatters.get(format)
        if not formatter:
            raise ValueError(f"Unknown format: {format}")
        
        return formatter.format(report)
    
    def _create_overview_section(
        self,
        results: List[EvaluationResult],
    ) -> ReportSection:
        """Create overview section."""
        content_parts = [f"This report contains {len(results)} evaluation run(s)."]
        
        total_samples = sum(r.sample_count for r in results)
        total_success = sum(r.success_count for r in results)
        
        content_parts.append(
            f"Total samples evaluated: {total_samples}, "
            f"Success rate: {total_success/total_samples*100:.1f}%"
        )
        
        return ReportSection(
            title="Overview",
            content=" ".join(content_parts),
            level=1,
        )
    
    def _create_metrics_section(
        self,
        results: List[EvaluationResult],
    ) -> ReportSection:
        """Create metrics section."""
        # Aggregate metrics
        aggregated = MetricAggregator.aggregate_results(results)
        
        # Create table
        rows = []
        for name, metric in sorted(aggregated.items()):
            rows.append([
                name,
                metric.formatted,
                f"±{metric.std:.3f}" if metric.std else "N/A",
                metric.type.value,
            ])
        
        table = {
            'headers': ['Metric', 'Mean', 'Std Dev', 'Type'],
            'rows': rows,
        }
        
        return ReportSection(
            title="Metrics Summary",
            content="Aggregated metrics across all evaluation runs.",
            tables=[table],
            level=1,
        )
    
    def _create_comparison_section(
        self,
        comparisons: List[ComparisonResult],
    ) -> ReportSection:
        """Create comparison section."""
        subsections = []
        
        for comp in comparisons:
            rows = []
            for name, diff in sorted(comp.metric_diffs.items()):
                status = "↑" if diff > 0 else "↓" if diff < 0 else "="
                rows.append([name, f"{diff:+.3f}", status])
            
            table = {
                'headers': ['Metric', 'Difference', 'Change'],
                'rows': rows,
            }
            
            subsections.append(ReportSection(
                title=f"{comp.baseline.name} vs {comp.comparison.name}",
                content=f"Improvements: {len(comp.improvements)}, Regressions: {len(comp.regressions)}",
                tables=[table],
                level=2,
            ))
        
        return ReportSection(
            title="Comparisons",
            content="Comparison of evaluation runs.",
            subsections=subsections,
            level=1,
        )
    
    def _generate_recommendations(
        self,
        results: List[EvaluationResult],
        comparisons: List[ComparisonResult],
    ) -> List[str]:
        """Generate recommendations."""
        recommendations = []
        
        if results:
            # Check for low metrics
            for result in results:
                for name, metric in result.metrics.items():
                    if metric.higher_is_better and metric.value < 0.5:
                        recommendations.append(
                            f"Consider improving {name} "
                            f"(current: {metric.formatted})"
                        )
        
        # Check comparisons for regressions
        for comp in comparisons:
            if comp.regressions:
                recommendations.append(
                    f"Address regressions in: {', '.join(comp.regressions)}"
                )
        
        return recommendations[:5]  # Limit to top 5
    
    def _generate_summary(
        self,
        results: List[EvaluationResult],
        comparisons: List[ComparisonResult],
    ) -> str:
        """Generate report summary."""
        parts = []
        
        if results:
            avg_metrics = MetricAggregator.aggregate_results(results)
            key_metrics = ['accuracy', 'precision', 'recall', 'f1']
            
            for metric_name in key_metrics:
                if metric_name in avg_metrics:
                    parts.append(
                        f"{metric_name}: {avg_metrics[metric_name].formatted}"
                    )
        
        if comparisons:
            total_improvements = sum(len(c.improvements) for c in comparisons)
            total_regressions = sum(len(c.regressions) for c in comparisons)
            parts.append(
                f"Changes: {total_improvements} improvements, "
                f"{total_regressions} regressions"
            )
        
        return " | ".join(parts) if parts else "No summary available."


# Convenience functions

def generate_report(
    results: List[Dict[str, Any]],
    title: str = "Evaluation Report",
    format: str = "html",
) -> str:
    """
    Quick report generation.
    
    Args:
        results: List of result dicts
        title: Report title
        format: Output format
        
    Returns:
        Formatted report string
    """
    # Convert dicts to EvaluationResults
    eval_results = []
    for r in results:
        metrics = {
            name: MetricValue(name=name, value=value)
            for name, value in r.get('metrics', {}).items()
        }
        eval_results.append(EvaluationResult(
            name=r.get('name', 'Evaluation'),
            metrics=metrics,
            sample_count=r.get('sample_count', 0),
            success_count=r.get('success_count', 0),
        ))
    
    reporter = EvaluationReporter()
    report = reporter.generate_report(eval_results, title=title)
    return reporter.format_report(report, format=format)


def compare_results(
    baseline: Dict[str, float],
    comparison: Dict[str, float],
) -> Dict[str, float]:
    """
    Quick result comparison.
    
    Args:
        baseline: Baseline metrics
        comparison: Comparison metrics
        
    Returns:
        Dict of metric differences
    """
    diffs = {}
    for name in set(baseline.keys()) | set(comparison.keys()):
        base_val = baseline.get(name, 0.0)
        comp_val = comparison.get(name, 0.0)
        diffs[name] = comp_val - base_val
    return diffs
