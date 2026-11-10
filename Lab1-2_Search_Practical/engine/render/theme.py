"""Visual identity of the dungeon renderer: palette, fonts, light sprites.

Everything is drawn procedurally: the repository ships zero image
files, yet the dungeon still gets stone, torchlight and fog.
"""

from __future__ import annotations

import pygame

# ------------------------------------------------------------- palette --
BG          = (11, 10, 18)
BG_DEEP     = (7, 6, 12)
STONE       = (88, 82, 105)
STONE_DARK  = (70, 64, 85)
GRAVEL      = (124, 112, 98)
MIRE        = (100, 80, 52)
MIRE_DARK   = (74, 58, 38)
WALL_TOP    = (106, 98, 130)
WALL_EDGE   = (140, 130, 168)
WALL_FACE   = (48, 42, 64)
GOLD        = (255, 215, 94)
GOLD_SOFT   = (255, 231, 158)
CYAN        = (110, 231, 255)
VIOLET      = (139, 92, 246)
VIOLET_SOFT = (167, 139, 250)
PORTAL_A    = (124, 245, 200)
PORTAL_B    = (56, 189, 248)
TORCH       = (255, 180, 84)
PARCHMENT   = (239, 227, 196)
INK         = (26, 22, 36)
DANGER      = (248, 113, 113)

# ------------------------------------------------------------ lighting --
# Lower values give a brighter, more legible maze. Torches, the hero and the
# gate still carve light out of the veil.
DARKNESS = 45      # alpha (0-255) of the dark veil over the whole view; was 134
VIGNETTE = 40      # strength of the dark frame fading towards the centre; was 110

# --------------------------------------------------------------- fonts --
_FONT_CHAIN = "dejavuserif,georgia,timesnewroman,serif"
_FONT_CHAIN_UI = "dejavusans,verdana,arial,sans"
_cache: dict = {}


def font(size: int, bold: bool = False, ui: bool = False) -> pygame.font.Font:
    key = ("f", size, bold, ui)
    if key not in _cache:
        chain = _FONT_CHAIN_UI if ui else _FONT_CHAIN
        _cache[key] = pygame.font.SysFont(chain, size, bold=bold)
    return _cache[key]


# --------------------------------------------------------------- lights --

def radial_light(radius: int, color=(255, 205, 130),
                 max_alpha: int = 160) -> pygame.Surface:
    """A soft radial gradient used to *subtract* darkness (torch glow)."""
    key = ("light", radius, color, max_alpha)
    if key not in _cache:
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            a = int(max_alpha * (1 - r / radius) ** 1.6)
            pygame.draw.circle(surf, (*color[:3], a), (radius, radius), r)
        _cache[key] = surf
    return _cache[key]


def glow_dot(radius: int, color, max_alpha: int = 120) -> pygame.Surface:
    """Additive halo for path runes / portal sparks."""
    key = ("glow", radius, tuple(color), max_alpha)
    if key not in _cache:
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            a = int(max_alpha * (1 - r / radius) ** 2)
            pygame.draw.circle(surf, (*color[:3], a), (radius, radius), r)
        _cache[key] = surf
    return _cache[key]


def vignette(size: tuple[int, int], strength: int = 110) -> pygame.Surface:
    """Dark frame that fades towards the centre (drawn once per size)."""
    key = ("vig", size, strength)
    if key not in _cache:
        w, h = size
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.fill((0, 0, 0, strength))
        hole = radial_light(int(max(w, h) * 0.62), (255, 255, 255),
                            max_alpha=strength)
        rect = hole.get_rect(center=(w // 2, h // 2))
        surf.blit(hole, rect, special_flags=pygame.BLEND_RGBA_SUB)
        _cache[key] = surf
    return _cache[key]


def font_mono(size: int, bold: bool = False) -> pygame.font.Font:
    key = ("m", size, bold)
    if key not in _cache:
        _cache[key] = pygame.font.SysFont(
            "dejavusansmono,consolas,menlo,couriernew,monospace",
            size, bold=bold)
    return _cache[key]


# syntax colours (unused in this lab)
CODE_DEFAULT = PARCHMENT
CODE_KEYWORD = GOLD
CODE_STRING  = (150, 219, 138)
CODE_COMMENT = (129, 119, 158)
CODE_NUMBER  = CYAN
CODE_DEFNAME = (126, 231, 196)


_DEFAULTS = {"CYAN": CYAN, "VIOLET": VIOLET}
_CB = {"CYAN": (238, 119, 51), "VIOLET": (68, 119, 170)}
PALETTE = "default"


def set_palette(name: str) -> None:
    """'default' or 'cb' (colour-blind safe: orange/blue)."""
    global CYAN, VIOLET, PALETTE
    src = _CB if name == "cb" else _DEFAULTS
    CYAN = src["CYAN"]
    VIOLET = src["VIOLET"]
    PALETTE = name

RUNE_NAMES = ("SUN", "MOON", "LEAF", "DROP", "FLAME",
              "STAR", "EYE", "KEY", "GATE")
