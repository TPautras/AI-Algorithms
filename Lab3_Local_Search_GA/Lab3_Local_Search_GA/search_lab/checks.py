"""Feasibility checks only. No neighbourhood, crossover, or search is supplied."""
import numpy as np
import pandas as pd


def canonical_cycle(tour):
    a = np.asarray(tour, dtype=int)
    a = np.roll(a, -int(np.flatnonzero(a == 0)[0]))
    forward = tuple(int(x) for x in a)
    return min(forward, (forward[0],) + forward[:0:-1])


def cycle_diversity(population):
    """Exact distinct-cycle fraction, with rotations/reversals identified."""
    if len(population) == 0:
        raise ValueError("The population is empty.")
    return len({canonical_cycle(tour) for tour in population}) / len(population)


def tour_report(problem, tour):
    """Audit the combinatorial edges, independently of crossings in a drawing."""
    if not problem.is_valid(tour):
        raise ValueError("Invalid tour: expected every city ID exactly once.")
    edges = problem.edges(tour)
    degree = np.bincount(np.asarray(edges).ravel(), minlength=problem.n)
    if not np.all(degree == 2):
        raise ValueError("Every city must have degree 2 in a closed TSP tour.")
    neighbors = {i: [] for i in range(problem.n)}
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    return pd.DataFrame([
        dict(city_id=i, city=problem.names[i], degree=int(degree[i]),
             neighbor_1=problem.names[neighbors[i][0]],
             neighbor_2=problem.names[neighbors[i][1]])
        for i in range(problem.n)
    ])


def check_neighbor(problem, neighbor, trials=100, require_change=True):
    """neighbor(tour, rng) -> a new valid anchored permutation."""
    rng = np.random.default_rng(731)
    for _ in range(trials):
        original = problem.random_tour(rng)
        saved = original.copy()
        candidate = neighbor(original, rng)
        assert np.array_equal(original, saved), "The neighbor function changed its input."
        assert problem.is_valid(candidate), "The neighbor is not a valid permutation."
        assert candidate[0] == 0, "Keep city 0 in the first position."
        if require_change:
            assert canonical_cycle(candidate) != canonical_cycle(original), "Same cycle / no-op."
    return f"Passed {trials} feasibility and input-copy checks."


def check_crossover(problem, crossover, trials=100):
    """crossover(parent_a, parent_b, rng) -> a new valid anchored permutation."""
    rng = np.random.default_rng(732)
    for _ in range(trials):
        a, b = problem.random_tour(rng), problem.random_tour(rng)
        before_a, before_b = a.copy(), b.copy()
        child = crossover(a, b, rng)
        assert problem.is_valid(child) and child[0] == 0, "Invalid child."
        assert np.array_equal(a, before_a) and np.array_equal(b, before_b), "Parent modified."
    return f"Passed {trials} crossover feasibility and input-copy checks."


def check_neighborhood(problem, neighbors, expected_size=None, trials=20, **options):
    """neighbors(tour, **options) -> list of all valid, distinct, anchored neighbours."""
    rng = np.random.default_rng(733)
    for _ in range(trials):
        original = problem.random_tour(rng)
        saved = original.copy()
        found = list(neighbors(original, **options))
        assert np.array_equal(original, saved), "The neighbourhood function changed its input."
        assert found, "The neighbourhood is empty."
        cycles = set()
        for candidate in found:
            assert problem.is_valid(candidate), "A neighbour is not a valid permutation."
            assert candidate[0] == 0, "Keep city 0 in the first position."
            cycles.add(canonical_cycle(candidate))
        assert canonical_cycle(original) not in cycles, "A neighbour is the same cycle as the input."
        assert len(cycles) == len(found), "Two neighbours describe the same cycle."
        if expected_size is not None:
            assert len(found) == expected_size, f"Expected {expected_size} neighbours, found {len(found)}."
    return f"Passed {trials} full-neighbourhood checks ({len(found)} distinct valid neighbours per tour)."
