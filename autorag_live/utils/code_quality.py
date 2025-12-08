"""
Code formatting and linting helper utilities.

Simple utilities to help maintain code quality and formatting standards
without requiring heavy IDE integration.

Example:
    >>> from autorag_live.utils.code_quality import check_function_length
    >>>
    >>> def my_func():
    ...     # function code
    ...     pass
    >>>
    >>> is_ok = check_function_length(my_func, max_lines=50)
"""

import ast
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


def check_function_length(func: Callable, max_lines: int = 50) -> bool:
    """
    Check if a function exceeds maximum line count.

    Args:
        func: Function to check
        max_lines: Maximum allowed lines

    Returns:
        True if function is within limit

    Example:
        >>> def small_func():
        ...     return 42
        >>> assert check_function_length(small_func, max_lines=10)
    """
    try:
        source = inspect.getsource(func)
        line_count = len(source.strip().split("\n"))
        return line_count <= max_lines
    except (OSError, TypeError):
        # Can't get source (built-in, etc.)
        return True


def count_code_lines(file_path: str, exclude_comments: bool = True) -> Dict[str, int]:
    """
    Count lines of code in a Python file.

    Args:
        file_path: Path to Python file
        exclude_comments: Whether to exclude comment-only lines

    Returns:
        Dictionary with line counts

    Example:
        >>> stats = count_code_lines("my_module.py")
        >>> print(f"Code lines: {stats['code']}")
    """
    path = Path(file_path)
    if not path.exists():
        return {"total": 0, "code": 0, "comments": 0, "blank": 0}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    blank = sum(1 for line in lines if not line.strip())
    comments = sum(1 for line in lines if line.strip().startswith("#"))
    code = total - blank - (comments if exclude_comments else 0)

    return {
        "total": total,
        "code": code,
        "comments": comments,
        "blank": blank,
    }


def find_long_functions(
    file_path: str,
    max_lines: int = 50,
) -> List[Tuple[str, int]]:
    """
    Find functions exceeding line count limit.

    Args:
        file_path: Path to Python file
        max_lines: Maximum allowed lines per function

    Returns:
        List of (function_name, line_count) tuples

    Example:
        >>> long_funcs = find_long_functions("module.py", max_lines=30)
        >>> for name, lines in long_funcs:
        ...     print(f"{name}: {lines} lines")
    """
    path = Path(file_path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    long_functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Calculate function length
            if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                if node.end_lineno is not None and node.lineno is not None:
                    func_length = node.end_lineno - node.lineno + 1
                    if func_length > max_lines:
                        long_functions.append((node.name, func_length))

    return long_functions


def check_cyclomatic_complexity(func: Callable) -> int:
    """
    Calculate approximate cyclomatic complexity.

    A rough measure based on decision points (if, for, while, except).

    Args:
        func: Function to analyze

    Returns:
        Complexity score (1 = simple, higher = more complex)

    Example:
        >>> def simple_func(x):
        ...     return x + 1
        >>> complexity = check_cyclomatic_complexity(simple_func)
        >>> assert complexity == 1
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        return 1

    complexity = 1  # Base complexity

    for node in ast.walk(tree):
        # Count decision points
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            # Each boolean operator adds complexity
            complexity += len(node.values) - 1

    return complexity


def suggest_refactoring(
    file_path: str,
    max_function_lines: int = 50,
    max_complexity: int = 10,
) -> List[str]:
    """
    Analyze code and suggest refactoring opportunities.

    Args:
        file_path: Path to Python file
        max_function_lines: Maximum function length
        max_complexity: Maximum cyclomatic complexity

    Returns:
        List of suggestion strings

    Example:
        >>> suggestions = suggest_refactoring("my_module.py")
        >>> for suggestion in suggestions:
        ...     print(f"- {suggestion}")
    """
    suggestions = []

    # Check for long functions
    long_funcs = find_long_functions(file_path, max_function_lines)
    for func_name, line_count in long_funcs:
        suggestions.append(
            f"Function '{func_name}' is {line_count} lines long. "
            f"Consider breaking it into smaller functions."
        )

    # Check line counts
    stats = count_code_lines(file_path)
    if stats["code"] > 500:
        suggestions.append(
            f"File has {stats['code']} lines of code. " "Consider splitting into multiple modules."
        )

    return suggestions


def analyze_module(file_path: str) -> Dict[str, Any]:
    """
    Comprehensive module analysis.

    Args:
        file_path: Path to Python file

    Returns:
        Dictionary with analysis results

    Example:
        >>> analysis = analyze_module("my_module.py")
        >>> print(f"Functions: {analysis['function_count']}")
        >>> print(f"Classes: {analysis['class_count']}")
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"error": "Syntax error in file"}

    # Count elements
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    line_stats = count_code_lines(file_path)

    return {
        "file": str(path),
        "function_count": len(functions),
        "class_count": len(classes),
        "import_count": len(imports),
        "line_stats": line_stats,
        "functions": [f.name for f in functions],
        "classes": [c.name for c in classes],
    }
