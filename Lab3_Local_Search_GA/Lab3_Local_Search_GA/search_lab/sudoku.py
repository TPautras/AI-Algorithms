"""Optional helpers for the Sudoku exercise (Section 7): grids and solutions, checks, figures, a success summary
and an interactive player. Use any of them, or none.

No representation, cost, move or search algorithm is supplied here. The figures and the player accept either a
model with a 9×9 `puzzle` array (0 = empty cell) or the 81-character puzzle string. run_search and benchmark accept
any problem that defines is_valid(state), cost(state), random_state(rng), neighbors(state, ...) and
random_neighbor(state, rng, ...).
"""
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import IFrame
from .visuals import STYLE, COLORS

# 0 = empty cell. Sources: Project Euler problem 96 (easy, medium), P. Norvig's 17-clue example (hard),
# A. Inkala (2012, presented by its author as the hardest Sudoku). Each grid has exactly one solution.
SUDOKU_PUZZLES = {
    "easy":   "003020600900305001001806400008102900700000008006708200002609500800203009005010300",
    "medium": "200080300060070084030500209000105408000000000402706000301007040720040060004010003",
    "hard":   "400000805030000000000700000020000060000080400000010000000603070500200000104000000",
    "inkala": "800000000003600000070090200050007000000045700000100030001000068008500010090000400",
}
# The unique solutions, for checks and diagnostics only (for instance: how far is a failed run from the solution?).
SUDOKU_SOLUTIONS = {
    "easy":   "483921657967345821251876493548132976729564138136798245372689514814253769695417382",
    "medium": "245981376169273584837564219976125438513498627482736951391657842728349165654812793",
    "hard":   "417369825632158947958724316825437169791586432346912758289643571573291684164875293",
    "inkala": "812753649943682175675491283154237896369845721287169534521974368438526917796318452",
}


def parse_sudoku(text):
    """A 9×9 integer array (0 = empty) from 81 characters ('0' or '.' for an empty cell) or from an array."""
    if isinstance(text, str):
        text = [0 if c in "0." else int(c) for c in text if c in "0123456789."]
    grid = np.asarray(text, dtype=int)
    if grid.size != 81 or grid.min() < 0 or grid.max() > 9:
        raise ValueError("A Sudoku has 81 cells with digits 0-9 (0 = empty).")
    return grid.reshape(9, 9)


def _puzzle_of(model_or_puzzle):
    return np.asarray(model_or_puzzle.puzzle) if hasattr(model_or_puzzle, "puzzle") else parse_sudoku(model_or_puzzle)


def _known_solution(puzzle):
    for name, text in SUDOKU_PUZZLES.items():
        if np.array_equal(parse_sudoku(text), puzzle):
            return name, parse_sudoku(SUDOKU_SOLUTIONS[name])
    return None, None


def conflict_cells(grid):
    """Mask of the filled cells whose digit appears twice or more in their row, column or box (display only)."""
    onehot = np.asarray(grid)[..., None] == np.arange(1, 10)                  # (row, column, digit)
    dup_row = onehot.sum(1, keepdims=True) > 1
    dup_col = onehot.sum(0, keepdims=True) > 1
    dup_box = np.repeat(np.repeat(onehot.reshape(3, 3, 3, 3, 9).sum((1, 3)) > 1, 3, 0), 3, 1)
    return (onehot & (dup_row | dup_col | dup_box)).any(-1)


# ------------------------------------------------------------------ checks
def check_sudoku_model(model, move="swap", trials=20):
    """For a model with puzzle, is_valid(grid), random_state(rng), cost(grid), neighbors(grid, move) and
    random_neighbor(grid, rng, move): validity, reproducibility, inputs never modified, cost 0 on the solution."""
    for name in ("random_state", "cost", "neighbors", "random_neighbor", "is_valid"):
        assert hasattr(model, name), f"The model needs a method {name}."
    a = model.random_state(np.random.default_rng(5))
    b = model.random_state(np.random.default_rng(5))
    assert np.array_equal(a, b), "random_state must depend only on the rng it receives."
    rng = np.random.default_rng(741)
    sizes, costs = set(), []
    for _ in range(trials):
        grid = model.random_state(rng)
        saved = np.array(grid, copy=True)
        assert model.is_valid(grid), "random_state must return a complete grid of digits 1-9 that keeps the clues."
        cost = model.cost(grid)
        assert np.isfinite(cost) and cost >= 0 and float(cost).is_integer(), "cost must be a non-negative integer."
        costs.append(float(cost))
        found = list(model.neighbors(grid, move))
        assert np.array_equal(grid, saved), "neighbors changed its input grid."
        assert found, "The neighbourhood is empty."
        keys = set()
        for candidate in found:
            assert model.is_valid(candidate), "A neighbour is not a valid grid (digits 1-9, clues kept)."
            assert not np.array_equal(candidate, grid), "A neighbour equals the current grid."
            keys.add(np.asarray(candidate, dtype=np.int64).tobytes())
        assert len(keys) == len(found), "Two neighbours are the same grid."
        sizes.add(len(found))
    assert min(costs) > 0, "A random grid has cost 0: the cost does not detect every conflict."
    name, solution = _known_solution(np.asarray(model.puzzle))
    solved = ""
    if solution is not None:
        assert model.cost(solution) == 0, f"The cost of the solution of '{name}' should be 0."
        solved = f", cost 0 on the solution of '{name}'"
    size = ", ".join(str(s) for s in sorted(sizes))
    return (f"Passed {trials} model checks: valid reproducible random states (mean cost {np.mean(costs):.1f}), "
            f"{size} distinct valid neighbours per state, inputs never modified{solved}.")


def check_sudoku_neighbor(model, neighbor, trials=200):
    """neighbor(grid, rng) -> a new valid grid, different from its input; the input is never modified."""
    rng = np.random.default_rng(742)
    for _ in range(trials):
        grid = model.random_state(rng)
        saved = np.array(grid, copy=True)
        candidate = neighbor(grid, rng)
        assert np.array_equal(grid, saved), "The function changed its input grid."
        assert model.is_valid(candidate), "The result is not a valid grid (digits 1-9, clues kept)."
        assert not np.array_equal(candidate, grid), "The result equals its input (no move)."
        assert not np.shares_memory(candidate, grid), "Return a new array."
    return f"Passed {trials} validity and input-copy checks."


def check_sudoku_crossover(model, crossover, trials=200):
    """crossover(parent_a, parent_b, rng) -> a new valid grid; the parents are never modified."""
    rng = np.random.default_rng(743)
    for _ in range(trials):
        a, b = model.random_state(rng), model.random_state(rng)
        saved_a, saved_b = np.array(a, copy=True), np.array(b, copy=True)
        child = crossover(a, b, rng)
        assert model.is_valid(child), "The child is not a valid grid (digits 1-9, clues kept)."
        assert np.array_equal(a, saved_a) and np.array_equal(b, saved_b), "A parent was modified."
        assert not np.shares_memory(child, a) and not np.shares_memory(child, b), "Return a new array."
    return f"Passed {trials} crossover validity and input-copy checks."


# ------------------------------------------------------------------ summary
def solve_summary(results):
    """For the table returned by benchmark: per method, runs, solved runs (best cost 0), solved fraction, median
    evaluations of the solved runs, median and quartiles of the final cost, median time."""
    rows = []
    for method, group in results.groupby("method", sort=False):
        solved = group.best_cost <= 0
        rows.append(dict(method=method, runs=len(group), solved=int(solved.sum()),
                         solved_fraction=float(solved.mean()),
                         median_evaluations_solved=float(group.evaluations[solved].median()) if solved.any() else np.nan,
                         median_cost=float(group.best_cost.median()),
                         q25_cost=float(group.best_cost.quantile(.25)), q75_cost=float(group.best_cost.quantile(.75)),
                         median_seconds=float(group.seconds.median())))
    return pd.DataFrame(rows).set_index("method").round(4)


# ------------------------------------------------------------------ figures
def _draw_grid(ax, puzzle, grid=None, previous=None, title=""):
    shown = puzzle if grid is None else np.asarray(grid)
    conflicts = conflict_cells(shown) if grid is not None else np.zeros((9, 9), bool)
    for r in range(9):
        for c in range(9):
            if conflicts[r, c]:
                ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#f6d2cc", zorder=1))
            if previous is not None and shown[r, c] != np.asarray(previous)[r, c]:
                ax.add_patch(plt.Rectangle((c + .06, r + .06), .88, .88, fill=False, edgecolor=COLORS[1], lw=2.2,
                                           zorder=3))
            if shown[r, c]:
                given = puzzle[r, c] > 0
                ax.text(c + .5, r + .54, str(int(shown[r, c])), ha="center", va="center", fontsize=12, zorder=4,
                        weight="bold" if given else "normal", color="#182b45" if given else COLORS[0])
    for k in range(10):
        lw = 2.2 if k % 3 == 0 else .6
        ax.plot([0, 9], [k, k], color="#182b45", lw=lw, zorder=2)
        ax.plot([k, k], [0, 9], color="#182b45", lw=lw, zorder=2)
    ax.set_xlim(-.05, 9.05), ax.set_ylim(9.05, -.05), ax.set_aspect("equal"), ax.axis("off")
    suffix = f" · {int(conflicts.sum())} cells in conflict" if grid is not None else ""
    ax.set_title(f"{title}{suffix}", loc="left", fontsize=10)


def show_sudoku(model_or_puzzle, grids=None, title=None):
    """Clues in bold, placed digits in teal, cells in conflict shaded, cells changed since the previous panel outlined.

    show_sudoku(model)                          the clues only
    show_sudoku(model, {"label": grid, ...})    one panel per grid
    show_sudoku({"name": puzzle, ...})          several puzzles side by side
    """
    if isinstance(model_or_puzzle, dict):
        panels = [(f"{name} · {int((parse_sudoku(p) > 0).sum())} clues", parse_sudoku(p), None, None)
                  for name, p in model_or_puzzle.items()]
    else:
        puzzle = _puzzle_of(model_or_puzzle)
        if grids is None:
            panels = [(f"{int((puzzle > 0).sum())} clues", puzzle, None, None)]
        else:
            items, panels, previous = list(grids.items()), [], None
            for label, grid in items:
                panels.append((label, puzzle, np.asarray(grid), previous))
                previous = np.asarray(grid)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.6), squeeze=False,
                                 layout="constrained")
        for ax, (label, puzzle, grid, previous) in zip(axes.ravel(), panels):
            _draw_grid(ax, puzzle, grid, previous, label)
        if title:
            fig.suptitle(title, fontsize=13, weight="bold")
        plt.close(fig)
    return fig


def plot_sudoku_trace(run):
    """Static evidence for one run of run_search (record=True): cost and best so far; temperature (mode "sa"),
    population diversity (mode "ga") or the Δ of the best neighbour at each step (mode "local")."""
    if not run.events:
        raise ValueError("Rerun with record=True.")
    events = run.events
    x = [e["evaluations"] for e in events]
    with plt.rc_context(STYLE):
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 3.7), layout="constrained")
        ax.plot(x, [e["cost"] for e in events], color=COLORS[0], lw=1, zorder=2,
                label="Population best" if run.mode == "ga" else "Current")
        ax.plot(x, [e["best"] for e in events], color=COLORS[2], linestyle="--", lw=2, zorder=3,
                label="Best evaluated so far")
        restarts = [e for e in events if e["note"] == "Random restart"]
        if restarts:
            ax.scatter([e["evaluations"] for e in restarts], [e["cost"] for e in restarts], marker="^", s=18,
                       color=COLORS[3], label="Random restart", zorder=4)
        if run.mode == "sa" and any(e["temperature"] is not None for e in events):
            bx.plot(x, [e["temperature"] for e in events], color=COLORS[1])   # every rise of the current curve is an accepted worse move
            bx.set(ylabel="Temperature", title="Temperature used in this run")
        elif run.mode == "ga":
            bx.plot(x, [e.get("diversity", np.nan) for e in events], color=COLORS[2])
            bx.set(ylabel="Population diversity", ylim=(0, 1.05), title="Diversity recorded in the trace")
        else:
            decisions = [e for e in events if e["accepted"] is not None]
            bx.scatter([e["evaluations"] for e in decisions], [e["delta"] for e in decisions], s=9,
                       c=[COLORS[0] if e["accepted"] else COLORS[1] for e in decisions])
            bx.axhline(0, color="#526379", lw=1)
            if run.mode == "sa":
                bx.set(ylabel="Candidate cost − current cost",
                       title="Δ of each proposal · teal accepted / orange rejected")
            else:
                bx.set(ylabel="Best neighbour cost − current cost",
                       title="Δ of the best neighbour · teal accepted (Δ = 0: sideways) / orange rejected")
        ax.set(ylabel="Cost (conflicts) ↓", title=f"{run.name} · {run.steps} steps")
        ax.legend(fontsize=8, frameon=False)
        for target in (ax, bx):
            target.set_xlabel("Objective evaluations")
            target.grid(alpha=.4)
        plt.close(fig)
    return fig


# ------------------------------------------------------------------ embedded player
def _flat(grid):
    return [int(v) for v in np.asarray(grid).ravel()]


def _player_document(model, run, max_frames=200):
    run.validate()
    if max_frames < 2:
        raise ValueError("Use max_frames >= 2.")
    if not run.frames:
        raise ValueError("Rerun your algorithm with record=True to create an animation.")
    events = run.events
    # Cumulative decision counts at every event, so that the player stays exact with sampled events.
    stats, acc = [], dict(accepted=0, decisions=0, worse_accepted=0, worse=0, sideways=0, restarts=0)
    for e in events:
        if e["accepted"] is not None:
            acc["decisions"] += 1
            acc["accepted"] += bool(e["accepted"])
            if e["delta"] is not None and e["delta"] > 0:
                acc["worse"] += 1
                acc["worse_accepted"] += bool(e["accepted"])
        acc["sideways"] += bool(e["accepted"]) and e["delta"] == 0
        acc["restarts"] += e["note"] == "Random restart"
        stats.append((e["evaluations"], dict(acc)))
    eval_keys = [s[0] for s in stats]

    def stats_at(evaluations):
        k = int(np.searchsorted(eval_keys, evaluations, side="right")) - 1
        return stats[max(k, 0)][1]

    frames = run.frames
    if len(frames) > max_frames:
        early = min(25, max_frames // 2)
        keep = sorted(set(range(early)) | set(np.linspace(early, len(frames) - 1, max_frames - early).astype(int)))
        frames = [frames[i] for i in keep]
    out = []
    for f in frames:
        item = {k: f[k] for k in ("evaluations", "cost", "best", "accepted", "delta", "temperature", "probability",
                                  "generation", "note")}
        item["grid"], item["best_grid"] = _flat(f["tour"]), _flat(f["best_tour"])
        if "candidate" in f:
            item["candidate"], item["candidate_cost"] = _flat(f["candidate"]), f["candidate_cost"]
        if "population_costs" in f:
            item["population_costs"] = f["population_costs"]
        if "diversity" in f:
            item["diversity"] = f["diversity"]
        item["stats"] = stats_at(f["evaluations"])
        out.append(item)
    if len(events) > 1500:                                   # The curve does not need every event.
        keep = np.unique(np.r_[np.linspace(0, len(events) - 1, 1500).astype(int), len(events) - 1])
        events = [events[i] for i in keep]
    curve = [dict(evaluations=e["evaluations"], cost=e["cost"], best=e["best"]) for e in events]
    payload = dict(name=run.name, seed=run.seed, mode=run.mode, puzzle=_flat(_puzzle_of(model)),
                   grid_name=getattr(model, "name", "sudoku"), frames=out, events=curve, total=run.evaluations,
                   steps=run.steps, settings=run.settings)
    assets = Path(__file__).with_name("assets")
    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    template = (assets / "sudoku_player.html").read_text(encoding="utf-8")
    return (template.replace("__STYLE__", (assets / "player.css").read_text(encoding="utf-8"))
            .replace("__DATA__", data)
            .replace("__SCRIPT__", (assets / "sudoku_player.js").read_text(encoding="utf-8")))


def animate_sudoku(model, run, max_frames=200):
    """Interactive player for one run of run_search (record=True) whose states are 9×9 grids
    (play/pause, stepping, speed, current/best/candidate grid). model: a model with `puzzle`, or the puzzle string."""
    document = _player_document(model, run, max_frames)
    return IFrame("about:blank", width="100%", height=860, extras=[
        'title="Interactive Sudoku search trace"', 'sandbox="allow-scripts"', 'style="border:0;background:#fff"',
        'srcdoc="' + html.escape(document, quote=True) + '"',
    ])


def save_sudoku_player(model, run, filename):
    path = Path(filename)
    path.write_text(_player_document(model, run), encoding="utf-8")
    return str(path.resolve())
