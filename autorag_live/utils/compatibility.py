"""Python version compatibility utilities."""
import sys
import warnings
from typing import Tuple


def check_python_version(minimum_version: Tuple[int, int] = (3, 10)) -> bool:
    """
    Check if the current Python version meets the minimum requirement.
    
    Args:
        minimum_version: Tuple of (major, minor) version numbers
        
    Returns:
        True if version is compatible, False otherwise
        
    Raises:
        RuntimeError: If Python version is too old
    """
    current_version = sys.version_info[:2]
    
    if current_version < minimum_version:
        version_str = f"{current_version[0]}.{current_version[1]}"
        required_str = f"{minimum_version[0]}.{minimum_version[1]}"
        
        error_msg = (
            f"AutoRAG-Live requires Python {required_str}+ but you are running "
            f"Python {version_str}. Please upgrade your Python installation."
        )
        
        raise RuntimeError(error_msg)
    
    return True


def warn_python_compatibility() -> None:
    """Issue a warning for potential Python compatibility issues."""
    current_version = sys.version_info[:2]
    
    if current_version < (3, 10):
        warnings.warn(
            "AutoRAG-Live is designed for Python 3.10+. "
            f"You are using Python {current_version[0]}.{current_version[1]}. "
            "Some features may not work correctly.",
            UserWarning,
            stacklevel=2
        )
    elif current_version >= (3, 13):
        warnings.warn(
            f"AutoRAG-Live has not been tested with Python {current_version[0]}.{current_version[1]}. "
            "Please report any issues you encounter.",
            UserWarning,
            stacklevel=2
        )


# Check version on import
try:
    check_python_version()
except RuntimeError as e:
    # Convert to warning for graceful degradation during development
    warnings.warn(str(e), UserWarning, stacklevel=2)