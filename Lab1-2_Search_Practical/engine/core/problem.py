"""Shared constants and plan replay for the maze engine.

The search interface taught in the notebook lives in ``catacombs_lab.models``:
``Problem`` (``actions``, ``result``, ``goal_test``, ``action_cost`` and the
cumulative ``path_cost``), ``Node`` and ``DungeonPathProblem``. This module only
keeps what the renderer and the run checks need: the direction table, the
``CUTOFF`` sentinel of depth-limited search, the expansion-budget exception and
``replay``, which walks a returned plan through a map to validate it.
"""

from __future__ import annotations

from .grid import DungeonMap

#: Sentinel returned by depth-limited search when the depth limit was hit
#: before the search space was exhausted.
CUTOFF = "cutoff"

#: Movement directions, in the canonical order actions are listed.
DIRECTIONS: dict[str, tuple[int, int]] = {
    "N": (0, -1),
    "E": (1, 0),
    "S": (0, 1),
    "W": (-1, 0),
}


class SearchBudgetExceeded(RuntimeError):
    """Raised when a search expands far more nodes than any correct
    algorithm would need — almost always the sign of an infinite loop
    (did you forget to track reached states?)."""


def replay(dungeon: DungeonMap, plan) -> dict:
    """Walk ``plan`` (a list of direction strings) through the dungeon.

    Returns a dict: ``ok`` (bool), ``reason`` (str), ``states`` (visited
    positions incl. start), ``length`` (number of moves), ``cost`` (total
    terrain cost), ``reached_goal`` (bool).
    """
    out = {"ok": False, "reason": "", "states": [], "length": 0,
           "cost": 0, "reached_goal": False}
    if plan is None:
        out["reason"] = "the search returned None (no plan found)"
        return out
    if not isinstance(plan, (list, tuple)):
        out["reason"] = f"expected a list of moves, got {type(plan).__name__}"
        return out

    pos = dungeon.start
    states = [pos]
    total = 0
    for i, action in enumerate(plan):
        if action not in DIRECTIONS:
            out["reason"] = f"move #{i + 1} is {action!r}, not one of N/E/S/W"
            out["states"] = states
            return out
        dx, dy = DIRECTIONS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not dungeon.passable(*nxt):
            out["reason"] = (f"move #{i + 1} ({action}) walks into a wall "
                             f"at {nxt}")
            out["states"] = states
            return out
        total += dungeon.terrain_cost(*nxt)
        pos = nxt
        states.append(pos)

    out["states"] = states
    out["length"] = len(plan)
    out["cost"] = total
    out["reached_goal"] = (pos == dungeon.goal)
    if not out["reached_goal"]:
        out["reason"] = f"the plan ends at {pos}, not at the gate {dungeon.goal}"
        return out
    out["ok"] = True
    out["reason"] = "the gate is reached"
    return out
