"""Timeline of a recorded search, ready to animate.

Phases:  EXPLORE  →  PATH  →  WALK  →  DONE
The playback holds *incremental* frontier updates so stepping forward
or restarting is cheap even for hundreds of expansions.
"""

from __future__ import annotations

EXPLORE, PATH, WALK, DONE = "explore", "path", "walk", "done"


class SearchPlayback:
    def __init__(self, events: list[dict], path_states: list[tuple]):
        self.events = events
        self.path_states = path_states          # includes the start
        self.reset()

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.phase = EXPLORE if self.events else (PATH if self.path_states
                                                  else DONE)
        self.k = 0.0                 # expansion cursor (float for smoothness)
        self.path_j = 0.0            # path reveal cursor
        self.walk_t = 0.0            # hero position along the path
        self.expanded: set = set()
        self.order: list = []          # expansion order, for the glow head
        self.frontier: set = set()
        if self.events:
            self.frontier = {self.events[0]["state"]}
        self._applied = 0

    # ------------------------------------------------------------------ #

    def _apply_until(self, k_int: int) -> None:
        while self._applied < k_int and self._applied < len(self.events):
            ev = self.events[self._applied]
            self.frontier.discard(ev["state"])
            if ev["state"] not in self.expanded:
                self.expanded.add(ev["state"])
                self.order.append(ev["state"])
            for child in ev["children"]:
                if child not in self.expanded:
                    self.frontier.add(child)
            self._applied += 1

    def update(self, dt: float, speed: float) -> None:
        """Advance the animation. ``speed`` is a user multiplier."""
        if self.phase == EXPLORE:
            self.k += dt * 26.0 * speed
            self._apply_until(int(self.k))
            if self._applied >= len(self.events):
                self.phase = PATH if self.path_states else DONE
        elif self.phase == PATH:
            self.path_j += dt * 34.0 * speed
            if self.path_j >= len(self.path_states):
                self.path_j = float(len(self.path_states))
                self.phase = WALK if len(self.path_states) > 1 else DONE
        elif self.phase == WALK:
            self.walk_t += dt * 7.0 * speed
            if self.walk_t >= len(self.path_states) - 1:
                self.walk_t = float(len(self.path_states) - 1)
                self.phase = DONE

    # ------------------------------------------------------------------ #

    def hero_pos(self) -> tuple[float, float]:
        """Hero grid position (floats) for the current phase."""
        if not self.path_states:
            return (0.0, 0.0)
        if self.phase in (EXPLORE, PATH):
            return tuple(map(float, self.path_states[0]))
        i = int(self.walk_t)
        frac = self.walk_t - i
        if i >= len(self.path_states) - 1:
            return tuple(map(float, self.path_states[-1]))
        (x0, y0), (x1, y1) = self.path_states[i], self.path_states[i + 1]
        return (x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac)

    def recent(self, n: int = 10) -> list[tuple]:
        """Last expanded cells with an age in [0, 1) — the wave's head."""
        if self.phase != EXPLORE or not self.order:
            return []
        tail = self.order[-n:]
        m = len(tail)
        return [(s, (m - 1 - i) / n) for i, s in enumerate(tail)]

    def visible_path(self) -> list[tuple]:
        j = int(self.path_j)
        return self.path_states[:j]

    @property
    def progress_label(self) -> str:
        return {EXPLORE: "exploring…", PATH: "binding the route…",
                WALK: "the hero walks", DONE: "done"}[self.phase]


ANNEAL = "anneal"
EVOLVE = "evolve"


class LocalPlayback:
    """Replays a local-search trace: every objective() call, in order.

    No hero, no path — the candidate placement *is* the picture, and
    the score curve is the story.
    """

    def __init__(self, trace, final_state, final_score,
                 max_frames: int = 700):
        frames = list(trace) or [(final_state, final_score)]
        step = max(1, len(frames) // max_frames)
        self.frames = frames[::step]
        if self.frames[-1][0] != final_state:
            self.frames.append((final_state, final_score))
        self.total_evals = len(trace)
        self.i = 0.0
        self.phase = ANNEAL
        self.best = max(s for _, s in self.frames)

    def update(self, dt: float, speed: float) -> None:
        if self.phase == ANNEAL:
            self.i += dt * speed * 110.0
            if self.i >= len(self.frames) - 1:
                self.i = float(len(self.frames) - 1)
                self.phase = DONE

    def frame(self):
        return self.frames[int(self.i)]

    def scores_so_far(self):
        return [s for _, s in self.frames[: int(self.i) + 1]]

    @property
    def progress_label(self) -> str:
        return "the stones seek their places…" if self.phase == ANNEAL \
            else "done"


GENS = "generations"
CHAMPION = "champion"


class EvoPlayback:
    """Replays a forge run: every evaluated child's landing cell and
    fitness, then the best homunculus walks its gauntlet."""

    def __init__(self, trace, best_states, best_fitness,
                 max_frames: int = 900):
        frames = list(trace) or [((0, 0), best_fitness, False)]
        step = max(1, len(frames) // max_frames)
        self.frames = frames[::step]
        self.total_evals = len(trace)
        self.best_states = list(best_states) or [(0, 0)]
        self.best = max(f for _, f, _ in self.frames + 
                        [((0, 0), best_fitness, False)])
        self.i = 0.0
        self.walk_t = 0.0
        self.phase = EVOLVE

    def update(self, dt: float, speed: float) -> None:
        if self.phase == EVOLVE:
            self.i += dt * speed * 140.0
            if self.i >= len(self.frames) - 1:
                self.i = float(len(self.frames) - 1)
                self.phase = WALK if len(self.best_states) > 1 else DONE
        elif self.phase == WALK:
            self.walk_t += dt * speed * 9.0
            if self.walk_t >= len(self.best_states) - 1:
                self.walk_t = float(len(self.best_states) - 1)
                self.phase = DONE

    def frame(self):
        _pos, fit, _r = self.frames[int(self.i)]
        return _pos, fit

    def scores_so_far(self):
        return [f for _, f, _ in self.frames[: int(self.i) + 1]]

    def recent(self, n: int = 60):
        k = int(self.i) + 1
        tail = self.frames[max(0, k - n): k]
        m = len(tail)
        return [(pos, (m - 1 - idx) / max(1, n), r)
                for idx, (pos, _f, r) in enumerate(tail)]

    def hero_pos(self):
        j = self.walk_t
        a = self.best_states[int(j)]
        b = self.best_states[min(int(j) + 1, len(self.best_states) - 1)]
        f = j - int(j)
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

    def walked(self):
        return self.best_states[: int(self.walk_t) + 1]

    @property
    def progress_label(self) -> str:
        return {EVOLVE: "the bloodline breeds…",
                WALK: "the firstborn walks",
                DONE: "done"}[self.phase]


LEARN = "learn"


class FrostPlayback:
    """Replays a winter: every training episode's return and ending
    cell, the Q-table crystallising through reported snapshots, then
    the frozen greedy policy walks the ice."""

    def __init__(self, episode_log, snapshots, walk,
                 max_frames: int = 600):
        frames = list(episode_log) or [(0.0, walk[0] if walk
                                        else (0, 0))]
        self.stride = max(1, len(frames) // max_frames)
        self.frames = frames[::self.stride]
        self.total_eps = len(episode_log)
        self.snapshots = list(snapshots)
        self.walk = list(walk) or [(0, 0)]
        self.best = max(r for r, _ in self.frames)
        self.i = 0.0
        self.walk_t = 0.0
        self.phase = LEARN if len(self.frames) > 1 else WALK

    def update(self, dt: float, speed: float) -> None:
        if self.phase == LEARN:
            self.i += dt * speed * 90.0
            if self.i >= len(self.frames) - 1:
                self.i = float(len(self.frames) - 1)
                self.phase = WALK if len(self.walk) > 1 else DONE
        elif self.phase == WALK:
            self.walk_t += dt * speed * 7.0
            if self.walk_t >= len(self.walk) - 1:
                self.walk_t = float(len(self.walk) - 1)
                self.phase = DONE

    def frame(self):
        ret, cell = self.frames[int(self.i)]
        return cell, ret

    def episode_now(self) -> int:
        if self.phase != LEARN:
            return self.total_eps
        return min(self.total_eps, int(self.i) * self.stride)

    def snapshot_now(self):
        ep = self.episode_now()
        chosen = None
        for mark, view in self.snapshots:
            if mark <= ep or chosen is None:
                chosen = view
            if mark > ep:
                break
        return chosen

    def scores_so_far(self):
        return [r for r, _ in self.frames[: int(self.i) + 1]]

    def recent(self, n: int = 40):
        k = int(self.i) + 1
        tail = self.frames[max(0, k - n): k]
        m = len(tail)
        return [(cell, (m - 1 - idx) / max(1, n), ret > 0)
                for idx, (ret, cell) in enumerate(tail)]

    def hero_pos(self):
        j = self.walk_t
        a = self.walk[int(j)]
        b = self.walk[min(int(j) + 1, len(self.walk) - 1)]
        f = j - int(j)
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

    def walked(self):
        return self.walk[: int(self.walk_t) + 1]

    @property
    def progress_label(self) -> str:
        return {LEARN: "the winters teach…",
                WALK: "the frozen policy walks",
                DONE: "done"}[self.phase]


SEEK = "seek"


class LockPlayback:
    """Replays a lock search: every (socket, colour, ok) try, then the
    solution settles and the leylines light."""

    def __init__(self, trace, solution, max_frames: int = 700,
                 describe=None):
        self.describe = describe
        frames = [(f + (None,))[:4] for f in (list(trace)
                                              or [(0, 0, True)])]
        self.scars: dict = {}
        for fr in frames:
            self.scars[fr[0]] = self.scars.get(fr[0], 0) + 1
        self.show_scars = False
        self.stride = max(1, len(frames) // max_frames)
        self.frames = frames[::self.stride]
        # small halls play slowly enough to read; big ones stay fluid
        self.rate = max(5.0, min(60.0, len(self.frames) / 6.0))
        self.total = len(trace)
        self.solution = solution or {}
        self.i = 0.0
        self.phase = SEEK if len(self.frames) > 1 else DONE
        self.provisional: dict = {}
        self._applied = 0

    def update(self, dt: float, speed: float) -> None:
        if self.phase != SEEK:
            return
        self.i += dt * speed * self.rate
        k = int(self.i)
        while self._applied <= min(k, len(self.frames) - 1):
            v, val, ok, _culprit = self.frames[self._applied]
            if ok:
                self.provisional[v] = val
            else:
                self.provisional.pop(v, None)   # visibly un-sealed
            self._applied += 1
        if self.i >= len(self.frames) - 1:
            self.i = float(len(self.frames) - 1)
            self.phase = DONE

    def frame(self):
        return self.frames[int(self.i)]

    def assigned(self) -> dict:
        return self.solution if self.phase == DONE else self.provisional

    def checks_so_far(self) -> int:
        return int((int(self.i) + 1) * self.stride)

    def scores_so_far(self):
        return list(range(0, self.checks_so_far(),
                          max(1, self.stride)))

    @property
    def best(self):
        return self.total

    @property
    def progress_label(self) -> str:
        if self.describe is not None:
            done = self.phase != SEEK
            return self.describe(*self.frame(), done=done)
        if self.phase != SEEK:
            return "the seal is complete"
        v, val, ok, _culprit = self.frame()
        return (f"cell {v} takes value {val}…" if ok
                else f"cell {v} tries value {val} — refused")
