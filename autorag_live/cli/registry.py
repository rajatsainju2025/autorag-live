"""CLI command registry pattern for reducing boilerplate."""

from typing import Callable, Dict, List, Optional

import typer


class CommandRegistry:
    """Registry for CLI commands with standardized help and error handling."""

    __slots__ = ("_app", "_commands", "_help_cache")

    def __init__(self, typer_app: typer.Typer):
        """Initialize command registry with a Typer app."""
        self._app = typer_app
        self._commands: Dict[str, Callable] = {}
        self._help_cache: Optional[str] = None

    def register(
        self,
        name: str,
        help_text: str,
        error_handler: Optional[Callable] = None,
    ) -> Callable:
        """
        Decorator to register a command.

        Args:
            name: Command name
            help_text: Help text for the command
            error_handler: Optional error handler function

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            # Store command reference
            self._commands[name] = func
            # Invalidate help cache
            self._help_cache = None

            # Add to Typer app
            @self._app.command(name=name, help=help_text)
            def wrapped_func(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if error_handler:
                        error_handler(name, e)
                    else:
                        typer.echo(f"❌ Error in {name}: {e!s}", err=True)
                        raise

            return wrapped_func

        return decorator

    def get_command(self, name: str) -> Optional[Callable]:
        """Get a registered command by name."""
        return self._commands.get(name)

    def list_commands(self) -> List[str]:
        """List all registered command names."""
        return list(self._commands.keys())

    def get_help_text(self) -> str:
        """Get formatted help text for all commands with caching."""
        if self._help_cache is not None:
            return self._help_cache

        lines = ["Available commands:\n"]
        for name in sorted(self._commands.keys()):
            lines.append(f"  {name}")
        self._help_cache = "\n".join(lines)
        return self._help_cache


# Example usage function
def create_command_registry(app: typer.Typer) -> CommandRegistry:
    """Factory function to create a command registry."""
    return CommandRegistry(app)
