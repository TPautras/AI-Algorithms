"""A small example graph with known search traces.

This fixture lets you check an implementation on a graph whose expected result
is known before running it on a maze. Nothing here performs a search.

* ``example_graph()`` : a directed graph with start S and goal G, with the
  expected traces of DFS, BFS, IDS and UCS (``EXAMPLE_EXPECTED``).
* ``EXAMPLE_POSITIONS`` : drawing coordinates for ``show_graph``.
* ``trace_graph(solver, problem)`` : run a solver on a ``GraphProblem`` and
  report the plan, its cost and the recorded goal-test and expansion orders.
"""
from __future__ import annotations

from .models import GraphProblem

# S is the start, G the goal; edges are directed, costs shown.
EXAMPLE_EDGES = {
    "S": {"A": 5, "B": 2, "C": 4},
    "A": {"D": 9, "E": 4},
    "B": {"G": 6},
    "C": {"F": 2},
    "D": {"H": 7},
    "E": {"G": 6},
    "F": {"G": 1},
    "G": {},
    "H": {},
}

# Drawing positions (x to the right, y upwards) used by show_graph.
EXAMPLE_POSITIONS = {"S": (0, 0), "A": (1, 1), "B": (1, 0), "C": (1, -1),
                     "D": (2, 1.6), "E": (2, .6), "F": (2, -1), "G": (3, 0), "H": (3, 1.6)}

# Expected results on that graph (goal tested when a node is selected).
EXAMPLE_EXPECTED = {
    "DFS": dict(tested=["S", "A", "D", "H", "E", "G"], path=["A", "E", "G"], cost=15),
    "BFS": dict(tested=["S", "A", "B", "C", "D", "E", "G"], path=["B", "G"], cost=8),
    "UCS": dict(tested=["S", "B", "C", "A", "F", "G"], path=["C", "F", "G"], cost=7),
    # IDS: the passes at limit 1 and limit 2 (limit 0 tests S only).
    "IDS": dict(tested_limit1=["S", "A", "B", "C"],
                tested_limit2=["S", "A", "D", "E", "B", "G"],
                path=["B", "G"], cost=8),
}


def example_graph():
    """The directed example graph (start S, goal G)."""
    return GraphProblem(EXAMPLE_EDGES, "S", "G", directed=True, positions=EXAMPLE_POSITIONS)


def trace_graph(solver, problem, *, heuristic=None, **options):
    """Run solver on a GraphProblem and report its trace.

    Returns a dict with the returned plan, its cost, the order of goal tests
    ("tested") and of successor requests ("expanded"). The recorded orders
    are cleared first, so the report describes this run only.
    """
    problem.reset()
    if heuristic is None:
        plan = solver(problem, **options)
    else:
        plan = solver(problem, heuristic, **options)
    cost = None
    if isinstance(plan, list):
        state, cost = problem.initial, 0
        for action in plan:
            child = problem.result(state, action)
            cost = problem.path_cost(cost, state, action, child)
            state = child
    return dict(plan=plan, cost=cost, tested=list(problem.tested), expanded=list(problem.expanded))
