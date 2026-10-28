"""Errors raised by the render subpackage."""

from litematic.parse.errors import LitematicError

__all__ = ["RenderError"]


class RenderError(LitematicError):
    """Raised when a render request cannot be fulfilled."""
