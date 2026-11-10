"""Problem and Node interfaces used throughout the lab, specialised for the maze.

These are data models and successor generation, not search algorithms.

The vocabulary of the problem formulation:

    ACTIONS(s)          -> Problem.actions(state)
    RESULT(s, a)        -> Problem.result(state, action)
    IS-GOAL(s)          -> Problem.goal_test(state)
    ACTION-COST(s,a,s') -> Problem.action_cost(state, action, next_state)

The lab also exposes the cumulative form ``path_cost(c, state1, action, state2)``,
which returns ``c + ACTION-COST(state1, action, state2)``. Both are kept: a
domain overrides ``action_cost`` (one step) and inherits ``path_cost``.
"""
from __future__ import annotations

from engine.core.grid import DungeonMap


class Problem:
    """The lab's problem interface. Subclasses describe a domain, not a strategy."""

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """ACTIONS(s): the legal actions in state, in a deterministic order."""
        raise NotImplementedError

    def result(self, state, action):
        """RESULT(s, a): the state reached by a legal action."""
        raise NotImplementedError

    def goal_test(self, state):
        """IS-GOAL(s). Accepts a single goal or a list of acceptable goals."""
        if isinstance(self.goal, list):
            return state in self.goal
        return state == self.goal

    def action_cost(self, state, action, next_state):
        """ACTION-COST(s, a, s'): the cost of ONE step. Default: every step costs 1."""
        return 1

    def path_cost(self, c, state1, action, state2):
        """Cumulative form: return the NEW total cost, c + ACTION-COST(state1, action, state2)."""
        return c + self.action_cost(state1, action, state2)

    def value(self, state):
        """Optimisation hook for later labs; unused in this route-finding lab."""
        raise NotImplementedError


class Node:
    """One path to a state, with the constructor and methods used in the notebook.

    A node keeps a state, a parent node, an action, a path cost (g) and a depth. Keep a queued node's path data fixed: a cheaper
    route to the same state gets a NEW Node. ``h`` and ``f`` are the only
    extra information needed for informed search.
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 if parent is None else parent.depth + 1
        self.h = 0

    def __repr__(self):
        return f"<Node {self.state}>"

    def __lt__(self, other):
        return self.state < other.state

    def __len__(self):
        return self.depth

    def expand(self, problem):
        """EXPAND(problem, node): one level of children, in the problem's action order."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """CHILD-NODE(problem, node, action): state, parent, action and g = g(parent) + cost."""
        state2 = problem.result(self.state, action)
        return Node(state2, self, action,
                    problem.path_cost(self.path_cost, self.state, action, state2))

    def path(self):
        """Return nodes from the root to this node, INCLUDING the root."""
        node, backwards = self, []
        while node is not None:
            backwards.append(node)
            node = node.parent
        return list(reversed(backwards))

    def solution(self):
        """Return actions only; the root contributes no action."""
        return [node.action for node in self.path()[1:]]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

    @property
    def f(self):
        """f(n) = g(n) + h(n)."""
        return self.path_cost + self.h


class DungeonPathProblem(Problem):
    """Maze formulation: a position is a state; a move pays for its destination."""

    DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

    def __init__(self, dungeon: DungeonMap):
        super().__init__(initial=dungeon.start, goal=dungeon.goal)
        self.dungeon = dungeon

    def actions(self, state):
        x, y = state
        return [action for action, (dx, dy) in self.DIRECTIONS.items()
                if self.dungeon.passable(x + dx, y + dy)]

    def result(self, state, action):
        dx, dy = self.DIRECTIONS[action]
        return state[0] + dx, state[1] + dy

    def action_cost(self, state, action, next_state):
        return self.dungeon.terrain_cost(*next_state)


class GraphProblem(Problem):
    """A road-map style graph (the directed example graph, the small reopening graphs, ...).

    ``edges`` maps a state to ``{neighbour: step cost}``. An action is the name of
    the neighbour it leads to, listed in the order given by ``edges``. With
    ``directed=False`` every road can be travelled both ways. The optional
    ``heuristic`` dictionary stores h(state) for informed search; the optional
    ``positions`` dictionary gives drawing coordinates to ``show_graph``.
    """

    def __init__(self, edges, initial, goal, *, directed=True, heuristic=None, positions=None):
        super().__init__(initial, goal)
        self.edges = {state: dict(neighbours) for state, neighbours in edges.items()}
        if not directed:
            for state, neighbours in list(self.edges.items()):
                for neighbour, cost in neighbours.items():
                    self.edges.setdefault(neighbour, {}).setdefault(state, cost)
        for state in list(self.edges):
            for neighbour in self.edges[state]:
                self.edges.setdefault(neighbour, {})
        self.heuristic = dict(heuristic or {})
        self.positions = dict(positions or {})   # optional (x, y) per state, for show_graph
        self.tested = []    # goal tests, in order
        self.expanded = []  # successor requests, in order

    def states(self):
        return list(self.edges)

    def actions(self, state):
        self.expanded.append(state)
        return list(self.edges.get(state, {}))

    def result(self, state, action):
        return action

    def goal_test(self, state):
        self.tested.append(state)
        return super().goal_test(state)

    def action_cost(self, state, action, next_state):
        return self.edges[state][next_state]

    def h(self, state, goal=None):
        """The stored estimate, usable as ``heuristic=problem.h`` for Greedy and A*."""
        return self.heuristic.get(state, 0)

    def reset(self):
        """Clear the recorded tested/expanded orders before another search."""
        self.tested.clear()
        self.expanded.clear()
        return self
