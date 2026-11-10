"""Dungeon scene renderer: draws everything.

One :class:`SceneRenderer` is prepared per dungeon; each frame it draws
the static stone, the living overlays of the search (violet = explored,
cyan = frontier, gold = the bound route), the hero, the portal, and the
torch-lit darkness on top.
"""

from __future__ import annotations

import math

import pygame

from ..core.grid import GOAL, HAZARD, START, WALL
from . import theme, tiles
from .animation import SearchPlayback

HUD_TOP = 66
HUD_BOTTOM = 46
SIDE_PANEL = 292


def _h(x: int, y: int) -> float:
    return ((x * 73856093) ^ (y * 19349663)) % 1000 / 1000.0


class SceneRenderer:
    def __init__(self, size: tuple[int, int] = (1280, 720)):
        self.size = size

    # ------------------------------------------------------------------ #

    def prepare(self, dungeon) -> None:
        self.dungeon = dungeon
        w, h = self.size
        avail_w = w - SIDE_PANEL - 70
        avail_h = h - HUD_TOP - HUD_BOTTOM - 40
        ts = int(min(avail_w / dungeon.width, avail_h / dungeon.height))
        self.ts = max(14, min(56, ts))
        self.ox = 35 + (avail_w - dungeon.width * self.ts) // 2
        self.oy = HUD_TOP + (h - HUD_TOP - HUD_BOTTOM
                             - dungeon.height * self.ts) // 2
        self.tileset = tiles.build_tileset(self.ts)
        self._build_static()

    def _build_static(self) -> None:
        d, ts = self.dungeon, self.ts
        surf = pygame.Surface(self.size)
        surf.fill(theme.BG)
        # faint outer glow behind the map
        pad = ts // 2
        frame = pygame.Rect(self.ox - pad, self.oy - pad,
                            d.width * ts + 2 * pad, d.height * ts + 2 * pad)
        pygame.draw.rect(surf, theme.BG_DEEP, frame, border_radius=12)
        pygame.draw.rect(surf, (30, 27, 44), frame, width=2, border_radius=12)

        for y in range(d.height):
            for x in range(d.width):
                r = self.cell_rect(x, y)
                ch = d.tile(x, y)
                if ch == WALL:
                    surf.blit(self.tileset["wall_top"], r)
                    if d.passable(x, y + 1):
                        face = pygame.transform.scale(
                            self.tileset["wall_face"], (ts, ts // 2))
                        surf.blit(face, (r.x, r.y + ts - ts // 2))
                elif ch == ",":
                    surf.blit(tiles.variant(self.tileset["gravel"], x, y), r)
                elif ch == "~":
                    surf.blit(tiles.variant(self.tileset["mire"], x, y), r)
                elif ch == HAZARD:
                    pygame.draw.rect(surf, (14, 8, 10), r)
                    pygame.draw.rect(surf, (34, 14, 14), r, width=2)
                    rng = (x * 7349 + y * 911) % 97
                    for k in range(3):
                        ex = r.x + 5 + ((rng + k * 31) % (ts - 10))
                        ey = r.y + 5 + ((rng * (k + 2)) % (ts - 10))
                        pygame.draw.circle(surf, (232, 96, 44), (ex, ey),
                                           max(2, ts // 12))
                        pygame.draw.circle(surf, (255, 176, 92), (ex, ey),
                                           max(1, ts // 22))
                else:  # ".", "S", "E"
                    surf.blit(tiles.variant(self.tileset["floor"], x, y), r)
                    if ch == START:
                        pygame.draw.rect(surf, (86, 78, 108),
                                         r.inflate(-ts * 0.42, -ts * 0.42),
                                         width=2, border_radius=3)
        self.static = surf
        # torch anchors: wall tiles whose south neighbour is floor
        self.torches = []
        for y in range(d.height):
            for x in range(d.width):
                if (d.tile(x, y) == WALL and d.passable(x, y + 1)
                        and _h(x, y) < 0.16):
                    self.torches.append((x, y))

    # ------------------------------------------------------------------ #

    def cell_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(self.ox + x * self.ts, self.oy + y * self.ts,
                           self.ts, self.ts)

    def cell_at(self, pos):
        """Screen position -> (x, y) cell, or None outside the map."""
        px, py = pos
        x = (px - self.ox) // self.ts
        y = (py - self.oy) // self.ts
        if 0 <= x < self.dungeon.width and 0 <= y < self.dungeon.height:
            return (int(x), int(y))
        return None

    def inspect_panel(self, target, info) -> None:
        (x, y), g, h = info
        f = None if g is None else g + h
        box = pygame.Rect(20, 90, 240, 96)
        pygame.draw.rect(target, (18, 15, 30), box, border_radius=10)
        pygame.draw.rect(target, theme.GOLD_SOFT, box, width=1,
                         border_radius=10)
        fnt = theme.font(15, ui=True)
        lines = [f"cell ({x}, {y})",
                 f"g*  true cost from start: "
                 f"{g if g is not None else '∞'}",
                 f"h   manhattan to gate:    {h}",
                 f"f = g* + h:               "
                 f"{f if f is not None else '∞'}"]
        for i, ln in enumerate(lines):
            target.blit(fnt.render(ln, True,
                                   theme.GOLD if i == 0
                                   else (196, 186, 214)),
                        (box.x + 14, box.y + 10 + i * 20))
        cx, cy = self.center(x, y)
        pygame.draw.rect(target, theme.GOLD,
                         self.cell_rect(x, y), width=2,
                         border_radius=4)

    def center(self, xf: float, yf: float) -> tuple[int, int]:
        return (int(self.ox + (xf + 0.5) * self.ts),
                int(self.oy + (yf + 0.5) * self.ts))

    # ------------------------------------------------------------------ #

    def draw(self, target: pygame.Surface, viz: SearchPlayback | None,
             t: float, orb: bool = False) -> None:
        ts = self.ts
        target.blit(self.static, (0, 0))

        hero = viz.hero_pos() if viz else tuple(map(float,
                                                    self.dungeon.start))

        self._darkness(target, hero, t)
        if orb:
            self._draw_orb_beam(target, hero, t)

        if viz is not None:
            lay = pygame.Surface(self.size, pygame.SRCALPHA)
            # explored — settled violet plates (search memory)
            for s in viz.expanded:
                r = self.cell_rect(*s).inflate(-3, -3)
                pygame.draw.rect(lay, (*theme.VIOLET, 108), r,
                                 border_radius=3)
                pygame.draw.rect(lay, (58, 40, 96, 170), r, width=1,
                                 border_radius=3)
            # frontier — bright cyan rings, breathing (discovered, waiting)
            for s in viz.frontier:
                pulse = 0.5 + 0.5 * math.sin(t * 5.0 + _h(*s) * 6.28)
                a = int(165 + 90 * pulse)
                r = self.cell_rect(*s).inflate(-ts * 0.22, -ts * 0.22)
                pygame.draw.rect(lay, (*theme.CYAN, a), r, width=3,
                                 border_radius=5)
            # the wave's head — the newest expansions flare gold
            for s, age in viz.recent(6):
                a = int(235 * (1.0 - age))
                r = self.cell_rect(*s).inflate(-5, -5)
                pygame.draw.rect(lay, (255, 214, 110, a), r, width=3,
                                 border_radius=4)
                cx, cy = r.center
                g = theme.glow_dot(max(4, ts // 4), theme.GOLD_SOFT,
                                   int(90 * (1.0 - age)))
                lay.blit(g, (cx - g.get_width() // 2,
                             cy - g.get_height() // 2))
            target.blit(lay, (0, 0))

            # the bound route — gold thread with a travelling spark
            path = viz.visible_path()
            if len(path) >= 1:
                pts = [self.center(*p) for p in path]
                for p in pts[::2]:
                    g = theme.glow_dot(max(3, ts // 3), theme.GOLD, 60)
                    target.blit(g, (p[0] - g.get_width() // 2,
                                    p[1] - g.get_height() // 2))
                if len(pts) >= 2:
                    pygame.draw.lines(target, theme.GOLD, False, pts,
                                      max(3, ts // 9))
                    for p in pts:
                        pygame.draw.circle(target, theme.GOLD_SOFT, p,
                                           max(2, ts // 12))
                    spark = pts[int(t * 6) % len(pts)]
                    g = theme.glow_dot(max(4, ts // 4), theme.GOLD_SOFT, 150)
                    target.blit(g, (spark[0] - g.get_width() // 2,
                                    spark[1] - g.get_height() // 2))

        self._draw_portal(target, t)
        self._draw_hero(target, hero, t)
        self._draw_flames(target, t)

    # ------------------------------------------------------------------ #

    def _draw_orb_beam(self, target, hero, t) -> None:
        """A faint dashed line from the hero to the gate."""
        x0, y0 = self.center(*hero)
        x1, y1 = self.center(*self.dungeon.goal)
        dx, dy = x1 - x0, y1 - y0
        dist = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / dist, dy / dist
        dash, gap = 9, 8
        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        pulse = 0.5 + 0.5 * math.sin(t * 2.2)
        col = (*theme.PORTAL_B, int(70 + 60 * pulse))
        offset = (t * 26) % (dash + gap)
        d = offset
        while d < dist:
            a = (x0 + ux * d, y0 + uy * d)
            b = (x0 + ux * min(dist, d + dash), y0 + uy * min(dist, d + dash))
            pygame.draw.line(lay, col, a, b, 2)
            d += dash + gap
        target.blit(lay, (0, 0))

    # ------------------------------------------------------------------ #

    def _draw_portal(self, target, t) -> None:
        gx, gy = self.dungeon.goal
        cx, cy = self.center(gx, gy)
        ts = self.ts
        pulse = 0.5 + 0.5 * math.sin(t * 2.4)
        g = theme.glow_dot(int(ts * (0.8 + 0.15 * pulse)), theme.PORTAL_B, 90)
        target.blit(g, (cx - g.get_width() // 2, cy - g.get_height() // 2))
        pygame.draw.circle(target, theme.BG_DEEP, (cx, cy), int(ts * 0.34))
        for i in range(3):
            a0 = t * (1.3 + 0.25 * i) + i * 2.09
            rect = pygame.Rect(0, 0, int(ts * (0.75 - 0.14 * i)),
                               int(ts * (0.75 - 0.14 * i)))
            rect.center = (cx, cy)
            col = theme.PORTAL_A if i % 2 == 0 else theme.PORTAL_B
            pygame.draw.arc(target, col, rect, a0, a0 + 3.6, 3)
        pygame.draw.circle(target, theme.PORTAL_A, (cx, cy),
                           max(2, int(ts * 0.07)))

    def _draw_hero(self, target, pos, t) -> None:
        ts = self.ts
        cx, cy = self.center(*pos)
        bob = math.sin(t * 3.1) * ts * 0.035
        cy = int(cy + bob - ts * 0.06)
        # shadow
        sh = pygame.Surface((ts, ts), pygame.SRCALPHA)
        pygame.draw.ellipse(sh, (0, 0, 0, 90),
                            (ts * 0.22, ts * 0.72, ts * 0.56, ts * 0.2))
        target.blit(sh, (cx - ts // 2, self.center(*pos)[1] - ts // 2))
        # cloak
        pts = [(-0.24, 0.34), (-0.19, -0.02), (0.0, -0.2),
               (0.19, -0.02), (0.24, 0.34)]
        cloak = [(cx + int(px * ts), cy + int(py * ts)) for px, py in pts]
        pygame.draw.polygon(target, (63, 48, 96), cloak)
        pygame.draw.polygon(target, theme.VIOLET_SOFT, cloak, width=1)
        # hood + face
        pygame.draw.circle(target, (63, 48, 96),
                           (cx, cy - int(ts * 0.18)), int(ts * 0.17))
        pygame.draw.circle(target, (236, 224, 200),
                           (cx, cy - int(ts * 0.15)), int(ts * 0.11))
        pygame.draw.circle(target, (63, 48, 96),
                           (cx, cy - int(ts * 0.22)), int(ts * 0.13), width=3)
        eye = max(1, ts // 20)
        for dx in (-1, 1):
            pygame.draw.circle(target, theme.INK,
                               (cx + dx * int(ts * 0.045),
                                cy - int(ts * 0.15)), eye)
        # staff with a gold tip
        x0, y0 = cx + int(ts * 0.2), cy + int(ts * 0.16)
        x1, y1 = cx + int(ts * 0.3), cy - int(ts * 0.34)
        pygame.draw.line(target, (120, 96, 66), (x0, y0), (x1, y1), 2)
        g = theme.glow_dot(max(3, ts // 6), theme.GOLD, 160)
        target.blit(g, (x1 - g.get_width() // 2, y1 - g.get_height() // 2))
        pygame.draw.circle(target, theme.GOLD_SOFT, (x1, y1),
                           max(1, ts // 16))

    def _quant(self, r: float) -> int:
        return max(8, int(r // 6) * 6)

    def _darkness(self, target, hero, t) -> None:
        dark = pygame.Surface(self.size, pygame.SRCALPHA)
        dark.fill((0, 0, 0, theme.DARKNESS))
        ts = self.ts

        def carve(px, py, radius, color, alpha):
            light = theme.radial_light(self._quant(radius), color, alpha)
            dark.blit(light, (px - light.get_width() // 2,
                              py - light.get_height() // 2),
                      special_flags=pygame.BLEND_RGBA_SUB)

        hx, hy = self.center(*hero)
        carve(hx, hy, ts * 3.4, (255, 214, 150), 150)
        gx, gy = self.center(*self.dungeon.goal)
        carve(gx, gy, ts * 2.6 * (1 + 0.08 * math.sin(t * 2.4)),
              (150, 230, 255), 120)
        for i, (x, y) in enumerate(self.torches):
            flick = 1 + 0.10 * math.sin(t * 9 + i * 1.7) \
                      + 0.05 * math.sin(t * 23 + i)
            px, py = self.center(x, y)
            carve(px, py + ts // 2, ts * 2.1 * flick, (255, 196, 120), 120)
        target.blit(dark, (0, 0))
        target.blit(theme.vignette(self.size, theme.VIGNETTE), (0, 0))

    def _draw_flames(self, target, t) -> None:
        ts = self.ts
        for i, (x, y) in enumerate(self.torches):
            px, py = self.center(x, y)
            py += int(ts * 0.18)
            # bracket
            pygame.draw.line(target, (70, 56, 44), (px, py + ts // 5),
                             (px, py), 3)
            fl = 1 + 0.22 * math.sin(t * 11 + i * 2.1)
            h1 = int(ts * 0.24 * fl)
            pygame.draw.circle(target, (255, 140, 60), (px, py - h1 // 2),
                               max(2, int(ts * 0.10)))
            pygame.draw.circle(target, theme.TORCH, (px, py - h1),
                               max(2, int(ts * 0.075)))
            pygame.draw.circle(target, (255, 236, 170),
                               (px, py - int(h1 * 1.25)),
                               max(1, int(ts * 0.04)))
            # rising ember
            ph = (t * 0.7 + i * 0.37) % 1.0
            ey = py - h1 - int(ph * ts * 0.8)
            ex = px + int(math.sin(t * 3 + i) * ts * 0.08)
            if ph < 0.85:
                pygame.draw.circle(target, (255, 200, 120), (ex, ey), 1)

    # ------------------------------------------------------------------ #
    #  HUD & panels                                                      #
    # ------------------------------------------------------------------ #

    def panel(self, target, rect, alpha=170, border=theme.GOLD) -> None:
        s = pygame.Surface(rect.size, pygame.SRCALPHA)
        s.fill((*theme.INK, alpha))
        pygame.draw.rect(s, (*border, 120), s.get_rect(), width=1,
                         border_radius=10)
        target.blit(s, rect)

    def hud(self, target, title: str, subtitle: str, right_lines: list[str],
            hint: str, legend: bool = False) -> None:
        w, _ = self.size
        bar = pygame.Surface((w, HUD_TOP), pygame.SRCALPHA)
        bar.fill((*theme.INK, 200))
        pygame.draw.line(bar, (*theme.GOLD, 130), (0, HUD_TOP - 1),
                         (w, HUD_TOP - 1))
        target.blit(bar, (0, 0))
        target.blit(theme.font(26, bold=True).render(title, True,
                                                     theme.GOLD), (26, 8))
        target.blit(theme.font(15).render(subtitle, True, theme.PARCHMENT),
                    (27, 40))
        for i, line in enumerate(right_lines[:3]):
            s = theme.font(14, ui=True).render(line, True, theme.PARCHMENT)
            target.blit(s, (w - s.get_width() - 24, 10 + i * 17))

        foot = pygame.Surface((w, HUD_BOTTOM), pygame.SRCALPHA)
        foot.fill((*theme.INK, 190))
        pygame.draw.line(foot, (60, 54, 82), (0, 0), (w, 0))
        target.blit(foot, (0, self.size[1] - HUD_BOTTOM))
        s = theme.font(14, ui=True).render(hint, True, (196, 186, 214))
        target.blit(s, ((w - s.get_width()) // 2,
                        self.size[1] - HUD_BOTTOM + 13))
        if legend:
            self._legend(target)

    def _legend(self, target) -> None:
        """Colour chips naming the four search states, bottom-left."""
        items = [((255, 214, 110), "expanding now"),
                 (theme.CYAN, "frontier"),
                 (theme.VIOLET, "explored"),
                 (theme.GOLD, "the route")]
        x = 22
        y = self.size[1] - HUD_BOTTOM + 12
        f = theme.font(13, ui=True)
        for col, label in items:
            pygame.draw.rect(target, col, (x, y + 2, 12, 12),
                             border_radius=3)
            pygame.draw.rect(target, (20, 18, 30), (x, y + 2, 12, 12),
                             width=1, border_radius=3)
            s = f.render(label, True, (196, 186, 214))
            target.blit(s, (x + 17, y))
            x += 17 + s.get_width() + 12

    def banner(self, target, text: str, color=theme.GOLD, sub: str = "") -> None:
        w, h = self.size
        f = theme.font(42, bold=True)
        surf = f.render(text, True, color)
        x = (w - surf.get_width()) // 2
        y = int(h * 0.16)
        pad = 26
        rect = pygame.Rect(x - pad, y - 14, surf.get_width() + 2 * pad,
                           surf.get_height() + (44 if sub else 24))
        self.panel(target, rect, alpha=205, border=color)
        shadow = f.render(text, True, theme.BG_DEEP)
        target.blit(shadow, (x + 2, y + 2))
        target.blit(surf, (x, y))
        if sub:
            s = theme.font(16).render(sub, True, theme.PARCHMENT)
            target.blit(s, ((w - s.get_width()) // 2,
                            y + surf.get_height() + 4))

    # ------------------------------------------------------------------ #
    #  Floor III — wardstone (local search) mode                         #
    # ------------------------------------------------------------------ #

    def draw_local(self, target, viz, ward, t: float) -> None:
        """Vault fully lit: coverage glow, stones, and the score curve."""
        ts = self.ts
        target.blit(self.static, (0, 0))
        state, score = viz.frame()

        counts: dict = {}
        for pos in state:
            for c in ward.coverage(pos):
                counts[c] = counts.get(c, 0) + 1

        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        for (x, y), n in counts.items():
            a = 40 + min(n, 3) * 30
            pygame.draw.rect(lay, (*theme.PORTAL_A, a),
                             self.cell_rect(x, y).inflate(-2, -2),
                             border_radius=3)
        for (x, y) in state:
            cx, cy = self.center(x, y)
            g = theme.glow_dot(int(ts * 0.9), theme.GOLD, 70)
            lay.blit(g, (cx - g.get_width() // 2, cy - g.get_height() // 2))
            pygame.draw.circle(lay, theme.GOLD, (cx, cy), int(ts * 0.30))
            pygame.draw.circle(lay, (60, 44, 10), (cx, cy), int(ts * 0.30),
                               width=2)
            pygame.draw.circle(lay, theme.GOLD_SOFT, (cx, cy),
                               max(2, int(ts * 0.10)))
        target.blit(lay, (0, 0))
        target.blit(theme.vignette(self.size, 90), (0, 0))
        self._draw_flames(target, t)
        self._score_curve(target, viz, score)

    def _score_curve(self, target, viz, score, label="score") -> None:
        w, h = self.size
        pw, ph = SIDE_PANEL - 40, 62
        rect = pygame.Rect(w - SIDE_PANEL + 20, HUD_TOP + 26, pw, ph)
        self.panel(target, rect, alpha=195)
        scores = viz.scores_so_far()
        if len(scores) >= 2:
            lo, hi = min(scores), max(viz.best, 1)
            span = max(1, hi - lo)
            pts = []
            n = len(scores)
            for i, s in enumerate(scores[-160:]):
                frac_x = i / max(1, min(n, 160) - 1)
                x = rect.x + 10 + frac_x * (pw - 20)
                y = rect.bottom - 10 - ((s - lo) / span) * (ph - 30)
                pts.append((x, y))
            if len(pts) >= 2:
                pygame.draw.lines(target, theme.CYAN, False, pts, 2)
            pygame.draw.circle(target, theme.GOLD, pts[-1], 3)
        s = theme.font(13, bold=True, ui=True).render(
            f"{label} {score} · best {viz.best:g}", True, theme.GOLD_SOFT)
        target.blit(s, (rect.x + 10, rect.y + 6))

    def _legend_local(self, target) -> None:
        items = [(theme.GOLD, "wardstone"),
                 (theme.PORTAL_A, "warded floor"),
                 (theme.CYAN, "score curve")]
        x = 22
        y = self.size[1] - HUD_BOTTOM + 12
        f = theme.font(13, ui=True)
        for col, label in items:
            pygame.draw.rect(target, col, (x, y + 2, 12, 12),
                             border_radius=3)
            s = f.render(label, True, (196, 186, 214))
            target.blit(s, (x + 17, y))
            x += 17 + s.get_width() + 12

    # ------------------------------------------------------------------ #
    #  Floor IV — evolution (genetic forge) mode                         #
    # ------------------------------------------------------------------ #

    def _fitness_curves(self, target, viz) -> None:
        w, h = self.size
        pw, ph = 260, 74
        rect = pygame.Rect(w - pw - 18, h - HUD_BOTTOM - ph - 12, pw, ph)
        self.panel(target, rect, alpha=195)
        means, bests = (viz.mean_best_curves() if viz.generations
                        else ([], []))
        series = [(means, theme.CYAN, 2), (bests, theme.GOLD, 2)]
        allv = (means + bests) or [0.0, 1.0]
        lo, hi = min(allv), max(max(allv), viz.best_fit)
        span = max(1e-6, hi - lo)
        for vals, col, wd in series:
            if len(vals) >= 2:
                pts = []
                for i, v in enumerate(vals[-140:]):
                    fx = i / max(1, min(len(vals), 140) - 1)
                    x = rect.x + 10 + fx * (pw - 20)
                    y = rect.bottom - 10 - ((v - lo) / span) * (ph - 34)
                    pts.append((x, y))
                pygame.draw.lines(target, col, False, pts, wd)
        s = theme.font(13, bold=True, ui=True).render(
            f"best {viz.best_fit:.0f} · gen "
            f"{viz.gen_index + 1}/{max(1, len(viz.generations))}",
            True, theme.GOLD_SOFT)
        target.blit(s, (rect.x + 10, rect.y + 6))

    def draw_evo(self, target, viz, t: float) -> None:
        ts = self.ts
        target.blit(self.static, (0, 0))
        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        # embers — where each evaluated child fell
        for pos, age, reached in viz.recent(60):
            cx, cy = self.center(*pos)
            fade = 1.0 - age
            if reached:
                pygame.draw.circle(lay, (*theme.GOLD, int(230 * fade)),
                                   (cx, cy), max(3, ts // 5), width=2)
                g = theme.glow_dot(max(4, ts // 3), theme.GOLD_SOFT,
                                   int(120 * fade))
            else:
                g = theme.glow_dot(max(3, ts // 4), (255, 120, 60),
                                   int(120 * fade))
            lay.blit(g, (cx - g.get_width() // 2,
                         cy - g.get_height() // 2))
        # the best bloodline walks
        walked = viz.walked() if viz.phase in ("walk", "done") else []
        if len(walked) >= 2:
            pts = [self.center(*p) for p in walked]
            pygame.draw.lines(lay, theme.GOLD, False, pts,
                              max(3, ts // 9))
        target.blit(lay, (0, 0))
        if viz.phase in ("walk", "done"):
            self._draw_portal(target, t)
            self._draw_hero(target, viz.hero_pos(), t)
        target.blit(theme.vignette(self.size, 80), (0, 0))
        self._draw_flames(target, t)
        _pos, fit = viz.frame()
        self._score_curve(target, viz, round(fit), label="fitness")

    def _legend_evo(self, target) -> None:
        items = [((255, 120, 60), "a child, tested"),
                 (theme.GOLD, "reached the altar"),
                 (theme.CYAN, "fitness curve")]
        x = 22
        y = self.size[1] - HUD_BOTTOM + 12
        f = theme.font(13, ui=True)
        for col, label in items:
            pygame.draw.rect(target, col, (x, y + 2, 12, 12),
                             border_radius=3)
            s = f.render(label, True, (196, 186, 214))
            target.blit(s, (x + 17, y))
            x += 17 + s.get_width() + 12

    # ------------------------------------------------------------------ #
    #  Floor V — frost (Q-learning) mode                                 #
    # ------------------------------------------------------------------ #

    _ARROWS = {"N": ((0, -1), (-1, 1), (1, 1)),
               "S": ((0, 1), (-1, -1), (1, -1)),
               "E": ((1, 0), (-1, -1), (-1, 1)),
               "W": ((-1, 0), (1, -1), (1, 1))}

    def draw_frost(self, target, viz, t: float) -> None:
        ts = self.ts
        target.blit(self.static, (0, 0))
        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        snap = viz.snapshot_now()
        if snap:
            vals = [v for v, _a in snap.values()]
            lo, hi = min(vals), max(vals)
            span = max(1e-9, hi - lo)
            for (x, y), (v, a) in snap.items():
                f = (v - lo) / span
                col = (int(70 + f * (255 - 70)),
                       int(120 + f * (206 - 120)),
                       int(210 + f * (120 - 210)))
                r = self.cell_rect(x, y).inflate(-3, -3)
                pygame.draw.rect(lay, (*col, 40 + int(70 * f)), r,
                                 border_radius=3)
                cx, cy = self.center(x, y)
                tip, l, rr = self._ARROWS[a]
                s = ts * 0.16
                pts = [(cx + tip[0] * s * 1.6, cy + tip[1] * s * 1.6),
                       (cx + l[0] * s, cy + l[1] * s),
                       (cx + rr[0] * s, cy + rr[1] * s)]
                pygame.draw.polygon(lay, (24, 20, 38, 230), pts)
        for cell, age, success in viz.recent(40):
            cx, cy = self.center(*cell)
            fade = 1.0 - age
            if success:
                pygame.draw.circle(lay, (*theme.GOLD, int(220 * fade)),
                                   (cx, cy), max(3, ts // 5), width=2)
            else:
                g = theme.glow_dot(max(3, ts // 4), (225, 95, 95),
                                   int(110 * fade))
                lay.blit(g, (cx - g.get_width() // 2,
                             cy - g.get_height() // 2))
        walked = viz.walked() if viz.phase in ("walk", "done") else []
        ghost = getattr(viz, "ghost_walk", None)
        if ghost and len(ghost) >= 2 and viz.phase in ("walk", "done"):
            pts = [self.center(*p) for p in ghost]
            for i in range(0, len(pts) - 1, 2):
                pygame.draw.line(lay, (*theme.CYAN, 170), pts[i],
                                 pts[i + 1], max(2, ts // 12))
        if len(walked) >= 2:
            pts = [self.center(*p) for p in walked]
            pygame.draw.lines(lay, theme.GOLD, False, pts,
                              max(3, ts // 9))
        target.blit(lay, (0, 0))
        if viz.phase in ("walk", "done"):
            self._draw_portal(target, t)
            self._draw_hero(target, viz.hero_pos(), t)
        target.blit(theme.vignette(self.size, 80), (0, 0))
        self._draw_flames(target, t)
        _cell, ret = viz.frame()
        self._score_curve(target, viz, round(ret), label="return")

    def _legend_frost(self, target) -> None:
        items = [((110, 160, 230), "learned value"),
                 ((24, 20, 38), "greedy policy"),
                 ((225, 95, 95), "an episode ends"),
                 (theme.GOLD, "altar reached")]
        x = 22
        y = self.size[1] - HUD_BOTTOM + 12
        f = theme.font(13, ui=True)
        for col, label in items:
            pygame.draw.rect(target, col, (x, y + 2, 12, 12),
                             border_radius=3)
            pygame.draw.rect(target, (90, 84, 110), (x, y + 2, 12, 12),
                             width=1, border_radius=3)
            s = f.render(label, True, (196, 186, 214))
            target.blit(s, (x + 17, y))
            x += 17 + s.get_width() + 12

    # ------------------------------------------------------------------ #
    #  Floor VI — lock (CSP) mode                                        #
    # ------------------------------------------------------------------ #

    RUNES = [theme.GOLD, theme.CYAN, (196, 120, 255), theme.PORTAL_A,
             (235, 140, 90), (240, 200, 90), (120, 200, 235),
             (230, 120, 170), (170, 230, 140)]

    def _rune_glyph(self, lay, cx, cy, s, val, col, width=0):
        """Nine distinct shapes so runes read even without colour."""
        w = width or max(2, s // 5)
        if val == 0:
            pygame.draw.circle(lay, col, (cx, cy), s, width=w)
        elif val == 1:
            pts = [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
            pygame.draw.polygon(lay, col, pts, width=w)
        elif val == 2:
            pygame.draw.rect(lay, col,
                             (cx - s, cy - s, 2 * s, 2 * s), width=w)
        elif val == 3:
            pts = [(cx, cy - s), (cx + s, cy), (cx, cy + s),
                   (cx - s, cy)]
            pygame.draw.polygon(lay, col, pts, width=w)
        elif val == 4:
            pygame.draw.line(lay, col, (cx - s, cy), (cx + s, cy), w)
            pygame.draw.line(lay, col, (cx, cy - s), (cx, cy + s), w)
        elif val == 5:
            for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
                pygame.draw.line(lay, col,
                                 (cx - s * dx, cy - s * dy),
                                 (cx + s * dx, cy + s * dy), w)
        elif val == 6:
            pts = [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
            pygame.draw.polygon(lay, col, pts, width=w)
        elif val == 7:
            pygame.draw.line(lay, col, (cx - s, cy - s),
                             (cx + s, cy + s), w)
            pygame.draw.line(lay, col, (cx - s, cy + s),
                             (cx + s, cy - s), w)
        else:
            pygame.draw.circle(lay, col, (cx, cy), s, width=w)
            pygame.draw.circle(lay, col, (cx, cy), max(2, s // 3))

    # -- shared table drawing helpers ---------------------------------- #

    def _table_geom(self, csp):
        R, C, bR, bC = csp.table
        xs = sorted({p[0] for p in csp.positions.values()})
        sp = (xs[1] - xs[0]) if len(xs) > 1 else 2
        cell = sp * self.ts
        x0 = self.ox + (csp.positions[0][0] - sp // 2) * self.ts
        y0 = self.oy + (csp.positions[0][1] - sp // 2) * self.ts
        return R, C, bR, bC, cell, x0, y0

    def table_cell_at(self, pos, csp):
        """Screen position -> table cell id, or None."""
        if not csp.table:
            return None
        R, C, _bR, _bC, cell, x0, y0 = self._table_geom(csp)
        c = (pos[0] - x0) // cell
        r = (pos[1] - y0) // cell
        if 0 <= r < R and 0 <= c < C:
            return int(r * C + c)
        return None

    def _unit_band(self, geom, v, culprit):
        R, C, bR, bC, cell, x0, y0 = geom
        r1, c1 = divmod(v, C)
        r2, c2 = divmod(culprit, C)
        if r1 == r2:
            return pygame.Rect(x0, y0 + r1 * cell, C * cell, cell)
        if c1 == c2:
            return pygame.Rect(x0 + c1 * cell, y0, cell, R * cell)
        if bR and bC:
            return pygame.Rect(x0 + (c1 // bC) * bC * cell,
                               y0 + (r1 // bR) * bR * cell,
                               bC * cell, bR * cell)
        return None

    def _table_base(self, lay, geom, csp, assigned, scars=None):
        R, C, bR, bC, cell, x0, y0 = geom
        slab = pygame.Rect(x0 - 10, y0 - 10,
                           C * cell + 20, R * cell + 20)
        pygame.draw.rect(lay, (26, 22, 40, 235), slab,
                         border_radius=12)
        pygame.draw.rect(lay, (150, 130, 190, 200), slab,
                         width=2, border_radius=12)
        if scars:
            top = max(scars.values())
            for v, n in scars.items():
                r, c = divmod(v, C)
                a = int(24 + 150 * math.log(1 + n) /
                        math.log(1 + top))
                sc = pygame.Surface((cell - 6, cell - 6),
                                    pygame.SRCALPHA)
                sc.fill((240, 70, 70, a))
                lay.blit(sc, (x0 + c * cell + 3, y0 + r * cell + 3))
        for v in csp.variables:
            r, c = divmod(v, C)
            rect = pygame.Rect(x0 + c * cell + 3, y0 + r * cell + 3,
                               cell - 6, cell - 6)
            given = v in csp.fixed
            if given and not scars:
                pygame.draw.rect(lay, (16, 13, 26, 255), rect,
                                 border_radius=6)
            val = assigned.get(v, csp.fixed.get(v))
            if val is not None:
                col = ((196, 186, 214) if given else
                       self.RUNES[val % len(self.RUNES)])
                s = max(6, int(cell * 0.26))
                self._rune_glyph(lay, rect.centerx, rect.centery,
                                 s, val, col)
            pygame.draw.rect(lay, (150, 130, 190, 70), rect,
                             width=1, border_radius=6)
        for c in range(C + 1):
            x = x0 + c * cell
            thick = 4 if (bC and c % bC == 0) or c in (0, C) else 1
            pygame.draw.line(lay, (150, 130, 190, 210),
                             (x, y0), (x, y0 + R * cell), thick)
        for r in range(R + 1):
            y = y0 + r * cell
            thick = 4 if (bR and r % bR == 0) or r in (0, R) else 1
            pygame.draw.line(lay, (150, 130, 190, 210),
                             (x0, y), (x0 + C * cell, y), thick)

    def rune_panel(self, target, n_colors):
        """Glyph -> name key, in the side column under the curve."""
        w, _h = self.size
        rows = min(n_colors, len(theme.RUNE_NAMES))
        pw = SIDE_PANEL - 40
        ph = 30 + rows * 22
        rect = pygame.Rect(w - SIDE_PANEL + 20, HUD_TOP + 104, pw, ph)
        self.panel(target, rect, alpha=195)
        f = theme.font(14, ui=True)
        target.blit(theme.font(13, bold=True, ui=True).render(
            "THE RUNES", True, theme.GOLD), (rect.x + 12, rect.y + 8))
        for i in range(rows):
            cy = rect.y + 34 + i * 22
            self._rune_glyph(target, rect.x + 22, cy, 7, i,
                             self.RUNES[i % len(self.RUNES)])
            target.blit(f.render(theme.RUNE_NAMES[i], True,
                                 (196, 186, 214)),
                        (rect.x + 40, cy - 8))

    def draw_lock(self, target, viz, csp, t: float) -> None:
        target.blit(self.static, (0, 0))
        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        assigned = viz.assigned()
        done = viz.phase == "done"

        if csp.table:
            geom = self._table_geom(csp)
            R, C, bR, bC, cell, x0, y0 = geom
            scars = (viz.scars if done and
                     getattr(viz, "show_scars", False) else None)
            if viz.phase == "seek":
                v, val, ok, culprit = viz.frame()
                if not ok and culprit is not None:
                    band = self._unit_band(geom, v, culprit)
                    if band:
                        bs = pygame.Surface(band.size, pygame.SRCALPHA)
                        bs.fill((240, 70, 70, 60))
                        lay.blit(bs, band.topleft)
            self._table_base(lay, geom, csp, assigned, scars=scars)
            if viz.phase == "seek":
                v, val, ok, culprit = viz.frame()
                r, c = divmod(v, C)
                rect = pygame.Rect(x0 + c * cell + 2,
                                   y0 + r * cell + 2,
                                   cell - 4, cell - 4)
                pulse = 2 + int(2 * abs(math.sin(t * 7)))
                col = (self.RUNES[val % len(self.RUNES)] if ok
                       else (245, 70, 70))
                pygame.draw.rect(lay, col, rect, width=2 + pulse,
                                 border_radius=6)
                s = max(6, int(cell * 0.26))
                self._rune_glyph(lay, rect.centerx, rect.centery, s,
                                 val, col if ok else (245, 70, 70))
                if not ok and culprit is not None:
                    r2, c2 = divmod(culprit, C)
                    crect = pygame.Rect(x0 + c2 * cell + 2,
                                        y0 + r2 * cell + 2,
                                        cell - 4, cell - 4)
                    pygame.draw.rect(lay, (245, 70, 70, 235), crect,
                                     width=3, border_radius=6)
        elif csp.rooms:
            fnt = theme.font(max(12, int(self.ts * 0.6)), bold=True)
            for v in csp.variables:
                x, y, w, hgt = csp.rooms[v]
                rect = pygame.Rect(self.ox + x * self.ts,
                                   self.oy + y * self.ts,
                                   w * self.ts, hgt * self.ts)
                if v in assigned:
                    col = self.RUNES[assigned[v] % len(self.RUNES)]
                    fill = pygame.Surface(rect.size, pygame.SRCALPHA)
                    fill.fill((*col, 78))
                    lay.blit(fill, rect.topleft)
                    pygame.draw.rect(lay, (*col, 200), rect, width=2)
                else:
                    pygame.draw.rect(lay, (120, 110, 150, 90), rect,
                                     width=1)

        target.blit(lay, (0, 0))
        target.blit(theme.vignette(self.size, 80), (0, 0))
        self._draw_flames(target, t)
        if done:
            self._draw_portal(target, t)
        self._score_curve(target, viz, viz.checks_so_far(),
                          label="checks")
        if csp.table:
            self.rune_panel(target, csp.n_colors)

    def draw_lock_hand(self, target, csp, hand, t: float) -> None:
        """HAND MODE — the player fills the table; conflicts glow."""
        target.blit(self.static, (0, 0))
        lay = pygame.Surface(self.size, pygame.SRCALPHA)
        geom = self._table_geom(csp)
        R, C, bR, bC, cell, x0, y0 = geom
        assigned = hand["assign"]
        conflicts = hand["conflicts"]
        for (a, b) in conflicts:
            band = self._unit_band(geom, a, b)
            if band:
                bs = pygame.Surface(band.size, pygame.SRCALPHA)
                bs.fill((240, 70, 70, 46))
                lay.blit(bs, band.topleft)
        self._table_base(lay, geom, csp, assigned)
        bad_cells = {v for pair in conflicts for v in pair}
        for v in bad_cells:
            r, c = divmod(v, C)
            rect = pygame.Rect(x0 + c * cell + 2, y0 + r * cell + 2,
                               cell - 4, cell - 4)
            pygame.draw.rect(lay, (245, 70, 70, 235), rect,
                             width=3, border_radius=6)
        if hand.get("last") is not None and \
                hand["last"] not in bad_cells:
            r, c = divmod(hand["last"], C)
            rect = pygame.Rect(x0 + c * cell + 2, y0 + r * cell + 2,
                               cell - 4, cell - 4)
            pulse = 2 + int(2 * abs(math.sin(t * 7)))
            pygame.draw.rect(lay, theme.GOLD, rect,
                             width=1 + pulse, border_radius=6)
        target.blit(lay, (0, 0))
        target.blit(theme.vignette(self.size, 80), (0, 0))
        self._draw_flames(target, t)
        if hand.get("lawful"):
            self._draw_portal(target, t)
        self.rune_panel(target, csp.n_colors)

    def _legend_lock(self, target) -> None:
        items = [(theme.GOLD, "placed rune"),
                 ((196, 186, 214), "carved given"),
                 ((240, 70, 70), "repeats in unit"),
                 (theme.PORTAL_A, "lawful")]
        x = 22
        y = self.size[1] - HUD_BOTTOM + 12
        f = theme.font(13, ui=True)
        for col, label in items:
            pygame.draw.rect(target, col, (x, y + 2, 12, 12),
                             border_radius=3)
            s = f.render(label, True, (196, 186, 214))
            target.blit(s, (x + 17, y))
            x += 17 + s.get_width() + 12
