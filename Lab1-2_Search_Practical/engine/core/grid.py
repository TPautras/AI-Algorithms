"""Dungeon map loading and terrain model.

Map files are plain text. Lines starting with ``@`` are metadata
(``@key: value``); every other non-empty line is a row of the dungeon grid.

Tile legend
-----------
``#``  wall (impassable)
``.``  stone floor          (cost 1)
``,``  gravel               (cost 2)
``~``  mire / deep mud      (cost 5)
``S``  the entrance, where the search starts (cost 1)
``E``  the exit gate, the goal (cost 1)
``x``  hazard, impassable in this lab (no supplied map uses it)

The *cost* of a move is the terrain cost of the tile you step **onto**.
"""

from __future__ import annotations

from pathlib import Path

WALL = "#"
HAZARD = "x"
TERRAIN_COST = {".": 1, ",": 2, "~": 5, "S": 1, "E": 1}
START = "S"
GOAL = "E"
VALID_TILES = set(TERRAIN_COST) | {WALL, HAZARD}


class DungeonMap:
    """An immutable rectangular dungeon grid with metadata."""

    def __init__(self, text: str, name: str = "unnamed"):
        self.meta: dict[str, str] = {}
        rows: list[str] = []
        for raw in text.splitlines():
            line = raw.rstrip("\n")
            if line.startswith("@"):
                key, _, value = line[1:].partition(":")
                self.meta[key.strip()] = value.strip()
            elif line.strip():
                rows.append(line)
        if not rows:
            raise ValueError(f"map '{name}' has no grid rows")

        width = max(len(r) for r in rows)
        self.rows: list[str] = [r.ljust(width, WALL) for r in rows]
        self.width = width
        self.height = len(self.rows)
        self.name = self.meta.get("name", name)

        self.start: tuple[int, int] | None = None
        self.goal: tuple[int, int] | None = None
        for y, row in enumerate(self.rows):
            for x, ch in enumerate(row):
                if ch not in VALID_TILES:
                    raise ValueError(
                        f"map '{name}': unknown tile {ch!r} at ({x}, {y})"
                    )
                if ch == START:
                    if self.start is not None:
                        raise ValueError(f"map '{name}' has two start tiles")
                    self.start = (x, y)
                elif ch == GOAL:
                    if self.goal is not None:
                        raise ValueError(f"map '{name}' has two goal tiles")
                    self.goal = (x, y)
        if self.start is None:
            raise ValueError(f"map '{name}' has no start tile 'S'")
        if self.goal is None:
            raise ValueError(f"map '{name}' has no goal tile 'E'")

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: str | Path) -> "DungeonMap":
        path = Path(path)
        return cls(path.read_text(encoding="utf-8"), name=path.stem)

    def tile(self, x: int, y: int) -> str:
        return self.rows[y][x]

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, x: int, y: int) -> bool:
        return (self.in_bounds(x, y)
                and self.rows[y][x] not in (WALL, HAZARD))

    def terrain_cost(self, x: int, y: int) -> int:
        ch = self.rows[y][x]
        if ch == START:
            ch = self.meta.get("start_on", ".")
        elif ch == GOAL:
            ch = self.meta.get("goal_on", ".")
        return TERRAIN_COST[ch]

    def meta_int(self, key: str) -> int | None:
        value = self.meta.get(key)
        return int(value) if value is not None else None

    def floor_positions(self) -> list[tuple[int, int]]:
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.passable(x, y)
        ]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DungeonMap {self.name!r} {self.width}x{self.height}>"
