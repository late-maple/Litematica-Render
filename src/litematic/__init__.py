"""litematic - read-only parser for Minecraft Litematica .litematic schematics.

Instances are not thread-safe: parse and query a Litematic from a single
thread or asyncio task at a time.
"""

from litematic.errors import LitematicError, LitematicFormatError, NbtError
from litematic.model import (
    BlockEntity,
    BlockState,
    Entity,
    Litematic,
    LitematicMetadata,
    Region,
    Vec3i,
)
from litematic.parser import load, loads

__version__ = "0.1.0"

__all__ = [
    "BlockEntity",
    "BlockState",
    "Entity",
    "Litematic",
    "LitematicError",
    "LitematicFormatError",
    "LitematicMetadata",
    "NbtError",
    "Region",
    "Vec3i",
    "load",
    "loads",
]
