"""Offline isometric rendering of parsed .litematic schematics.

Instances are not thread-safe, matching the parse API: render one
Litematic at a time from a single thread or asyncio task.
"""

from litematic.render.errors import RenderError

__all__ = ["RenderError"]
