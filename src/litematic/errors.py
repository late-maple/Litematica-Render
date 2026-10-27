"""Public exception tree for litematic."""


class LitematicError(Exception):
    """Base class for all exceptions raised by this library."""


class NbtError(LitematicError):
    """Input is not valid Java NBT: bad tag type, truncated stream, bad compression."""


class LitematicFormatError(LitematicError):
    """Input is valid NBT but violates the .litematic structure constraints."""
