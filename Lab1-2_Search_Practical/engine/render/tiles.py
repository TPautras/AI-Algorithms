"""Procedural dungeon tiles.

Each tile is drawn once at load time into small Surfaces and then
blitted; per-tile variety comes from a deterministic hash of the tile
coordinates, so a given map always looks the same.
"""

from __future__ import annotations

import random

import pygame

from . import theme


def _shade(color, delta):
    return tuple(max(0, min(255, c + delta)) for c in color)


def _speckle(surf, rng, n, colors):
    w, h = surf.get_size()
    for _ in range(n):
        x, y = rng.randrange(w), rng.randrange(h)
        c = rng.choice(colors)
        surf.fill(c, (x, y, max(1, w // 24), max(1, w // 24)))


def build_tileset(ts: int, seed: int = 3) -> dict:
    """Return dict of pre-rendered tile surfaces for tile size ``ts``."""
    rng = random.Random(seed)
    tiles: dict = {"floor": [], "gravel": [], "mire": []}

    # -- stone floor variants ----------------------------------------- #
    for i in range(5):
        s = pygame.Surface((ts, ts))
        base = _shade(theme.STONE, rng.randint(-7, 7))
        s.fill(base)
        _speckle(s, rng, ts // 3, [_shade(base, -12), _shade(base, 10)])
        # flagstone seams
        seam = _shade(base, -18)
        if i % 2 == 0:
            pygame.draw.line(s, seam, (0, ts - 1), (ts, ts - 1))
            pygame.draw.line(s, seam, (ts - 1, 0), (ts - 1, ts))
        if rng.random() < 0.35:                      # crack
            x0 = rng.randrange(ts // 4, 3 * ts // 4)
            pts = [(x0, rng.randrange(2, ts // 3))]
            for _ in range(3):
                px, py = pts[-1]
                pts.append((px + rng.randint(-ts // 5, ts // 5),
                            py + rng.randrange(2, ts // 3)))
            pygame.draw.lines(s, _shade(base, -26), False, pts, 1)
        tiles["floor"].append(s)

    # -- gravel -------------------------------------------------------- #
    for _ in range(4):
        s = pygame.Surface((ts, ts))
        base = _shade(theme.GRAVEL, rng.randint(-6, 6))
        s.fill(base)
        _speckle(s, rng, ts, [_shade(base, -20), _shade(base, 16),
                              _shade(base, -34)])
        tiles["gravel"].append(s)

    # -- mire / mud ----------------------------------------------------- #
    for _ in range(4):
        s = pygame.Surface((ts, ts))
        base = _shade(theme.MIRE, rng.randint(-6, 6))
        s.fill(base)
        for _ in range(3):                            # wet blobs
            r = rng.randrange(ts // 6, ts // 3)
            cx, cy = rng.randrange(ts), rng.randrange(ts)
            pygame.draw.circle(s, _shade(theme.MIRE_DARK, rng.randint(-6, 6)),
                               (cx, cy), r)
        pygame.draw.circle(s, _shade(base, 14),
                           (rng.randrange(ts), rng.randrange(ts)),
                           max(1, ts // 10))
        tiles["mire"].append(s)

    # -- wall top & face ------------------------------------------------ #
    top = pygame.Surface((ts, ts))
    top.fill(theme.WALL_TOP)
    pygame.draw.rect(top, theme.WALL_EDGE, (0, 0, ts, 2))
    pygame.draw.rect(top, _shade(theme.WALL_TOP, -18), (0, ts - 2, ts, 2))
    _speckle(top, rng, ts // 4, [_shade(theme.WALL_TOP, -10),
                                 _shade(theme.WALL_TOP, 8)])
    tiles["wall_top"] = top

    face = pygame.Surface((ts, ts))
    face.fill(theme.WALL_FACE)
    brick = _shade(theme.WALL_FACE, -12)
    hi = _shade(theme.WALL_FACE, 14)
    rows = 3
    bh = ts // rows
    for r in range(rows):
        y = r * bh
        pygame.draw.line(face, brick, (0, y), (ts, y))
        off = (ts // 2) if r % 2 else 0
        for x in range(off, ts + 1, ts):
            pygame.draw.line(face, brick, (x % ts, y), (x % ts, y + bh))
    pygame.draw.rect(face, hi, (0, 0, ts, 1))
    tiles["wall_face"] = face

    return tiles


def variant(tiles: list, x: int, y: int) -> pygame.Surface:
    """Deterministic per-coordinate pick."""
    return tiles[(x * 928371 + y * 689287 + 12345) % len(tiles)]
