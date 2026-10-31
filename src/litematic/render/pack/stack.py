"""Pack stack: ordered zip packs, lookups scan top-down with first hit winning."""

import zipfile
from pathlib import Path
from typing import Sequence

from litematic.render.errors import RenderError


class PackStack:
    """Zip packs in lookup order: index 0 is checked first, the base is last."""

    def __init__(
        self, packs: list[tuple[Path, zipfile.ZipFile, set[str]]]
    ) -> None:
        self._packs = packs
        self._names: set[str] = set()
        for _, _, names in packs:
            self._names |= names

    def names(self) -> set[str]:
        """Union of entry names across every pack."""
        return self._names

    def exists(self, path: str) -> bool:
        return path in self._names

    def read(self, path: str) -> bytes | None:
        found = self.read_ranked(path)
        return None if found is None else found[0]

    def read_ranked(self, path: str) -> tuple[bytes, int] | None:
        """First pack holding path; returns (bytes, pack index)."""
        for index in range(len(self._packs)):
            _, archive, names = self._packs[index]
            if path in names:
                return archive.read(path), index
        return None

    def mcmeta(self, path: str, pack_index: int) -> bytes | None:
        """Texture metadata from the top down to the texture's pack, inclusive."""
        meta = f"{path}.mcmeta"
        for index in range(pack_index + 1):
            _, archive, names = self._packs[index]
            if meta in names:
                return archive.read(meta)
        return None

    def close(self) -> None:
        for _, archive, _ in self._packs:
            archive.close()

    def __enter__(self) -> "PackStack":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_stack(sources: Sequence[str | Path]) -> PackStack:
    """Open every zip; sources are [base, *add-ons] in user priority order."""
    opened: list[tuple[Path, zipfile.ZipFile, set[str]]] = []
    try:
        for position, source in enumerate(sources):
            path = Path(source)
            try:
                archive = zipfile.ZipFile(path)
            except (zipfile.BadZipFile, OSError) as exc:
                raise RenderError(
                    f"extract: not a resource pack: {path}: {exc}"
                ) from exc
            names = set(archive.namelist())
            opened.append((path, archive, names))
            if position and not any(name.startswith("assets/") for name in names):
                raise RenderError(
                    f"extract: not a resource pack: {path}: no assets/ entries"
                )
    except BaseException:
        for _, archive, _ in opened:
            archive.close()
        raise
    # Lookup order: earlier add-on wins, the base is always checked last.
    return PackStack([*opened[1:], opened[0]])
