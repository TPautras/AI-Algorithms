"""Fixed destination costs shared by search, checks and the dungeon movie.

No solver lives here. A cost function takes one (x, y) destination and returns
a strictly positive, finite entry cost. It is evaluated once per walkable cell.
"""
from copy import deepcopy
import math
import random

from engine.core.grid import DungeonMap


def unit_cost(state):
    """Every move costs 1, including entry into the goal."""
    return 1


def random_costs(maze, *, seed=7, low=1, high=9):
    """Return a fixed callable cost table; bounds are inclusive.

    Draw once, before searching. Repeated calls for a cell return the same
    number, independently of search order and of Python's global RNG.
    """
    from .core import load_map
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (seed, low, high)):
        raise ValueError('seed, low and high must be integers.')
    if not 1 <= low <= high:
        raise ValueError('Choose positive bounds with low <= high.')
    dungeon = load_map(maze)
    rng = random.Random(seed)
    values = {state: rng.randint(low, high) for state in dungeon.floor_positions()}

    def cost(state):
        return values[tuple(state)]

    cost.__name__ = f'random {low}–{high}, seed {seed}'
    return cost


class _CostedDungeon(DungeonMap):
    """A separate map snapshot; leave the input map and its costs untouched."""
    def __init__(self, dungeon, cost_function):
        self.__dict__.update(deepcopy(dungeon.__dict__))
        self.entry_costs = {}
        for state in self.floor_positions():
            raw = cost_function(state)
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Entry cost at {state} must be a positive finite number.') from exc
            if isinstance(raw, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Entry cost at {state} must be positive and finite; got {raw!r}.')
            self.entry_costs[state] = value
        self.cost_label = getattr(cost_function, '__name__', 'custom entry costs')

    def terrain_cost(self, x, y):
        return self.entry_costs[(x, y)]


def with_costs(maze, cost_function):
    """Freeze cost_function((x,y)) on a copy of a named or custom maze.

    The initial path costs zero. Each move pays its destination's fixed cost.
    Reuse the returned map to compare approaches on exactly the same costs.
    """
    from .core import load_map
    if not callable(cost_function):
        raise TypeError('cost_function must accept a destination (x, y).')
    return _CostedDungeon(load_map(maze), cost_function)
