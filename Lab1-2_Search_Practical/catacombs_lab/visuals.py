"""Readable dungeon maps and self-contained, inspectable search replays.

Rendering never runs a search. Frame order and frontier order come from the
algorithm, so the animation displays the recorded evidence without inventing
intermediate states.
"""
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import numpy as np
from IPython.display import IFrame


INK = "#18283d"
EXPLORED = "#8b5cf6"
FRONTIER = "#008ca8"
GOLD = "#f3bd55"
TERRAIN = {"#": "#253345", "x": "#55313d", ".": "#eef0eb",
           ",": "#d9c5a0", "~": "#95bdce", "S": "#eef0eb", "E": "#eef0eb"}
STYLE = {"font.family": "DejaVu Sans", "font.size": 10,
         "text.color": INK, "axes.labelcolor": INK, "axes.titleweight": "bold",
         "figure.facecolor": "white", "axes.facecolor": "#f6f8fb",
         "axes.spines.top": False, "axes.spines.right": False,
         "axes.edgecolor": "#d4dde7", "grid.color": "#dfe5ed"}


def _number(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):,.0f}" if float(value).is_integer() else f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _terrain_name(tile):
    return {"#": "Wall", "x": "Hazard (blocked)", ".": "Stone floor",
            ",": "Gravel", "~": "Mire", "S": "Entrance", "E": "Exit"}.get(tile, "Floor")


def _draw_map(ax, dungeon, frame=None, final_path=None, title=None, annotate=True):
    """Draw coordinates (x, y), with row 0 at the top of every view."""
    for y, row in enumerate(dungeon.rows):
        for x, tile in enumerate(row):
            ax.add_patch(Rectangle((x-.5, y-.5), 1, 1,
                                  facecolor=TERRAIN.get(tile, TERRAIN["."]),
                                  edgecolor="#ffffff", linewidth=.35, zorder=0))
            if annotate and tile in (",", "~", "x"):
                label = "×" if tile == "x" else _number(dungeon.terrain_cost(x, y))
                ax.text(x, y, label, ha="center", va="center", fontsize=7,
                        color="white" if tile == "x" else "#466078", zorder=1)
    frame = frame or {}
    explored = [tuple(p) for p in frame.get("explored", [])]
    frontier = [tuple(e["state"]) for e in frame.get("frontier", []) if not e.get("stale", False)]
    if explored:
        xy = np.asarray(explored)
        ax.scatter(xy[:, 0], xy[:, 1], color=EXPLORED, marker="s", s=47,
                   alpha=.60, linewidths=0, zorder=2)
    if frontier:
        xy = np.asarray(frontier)
        ax.scatter(xy[:, 0], xy[:, 1], facecolor=FRONTIER, edgecolor="white",
                   marker="D", s=35, linewidth=.55, zorder=3)
    path = list(final_path) if final_path is not None else frame.get("path", [])
    if path:
        xy = np.asarray(path)
        ax.plot(xy[:, 0], xy[:, 1], color=INK, linewidth=4.3, alpha=.75, zorder=4)
        ax.plot(xy[:, 0], xy[:, 1], color=GOLD, linewidth=2.8,
                linestyle="-" if final_path is not None else "--", zorder=5)
    current = frame.get("current")
    if current is not None:
        ax.scatter(*current, color=GOLD, edgecolor=INK, linewidth=1.5, s=105, zorder=6)
    for point, label in ((dungeon.start, "S"), (dungeon.goal, "E")):
        ax.text(*point, label, color=INK, ha="center", va="center", fontsize=9,
                weight="bold", zorder=8,
                bbox=dict(boxstyle="round,pad=.16", facecolor="white", edgecolor=INK, linewidth=.7))
    ax.set(xlim=(-.6, dungeon.width-.4), ylim=(dungeon.height-.4, -.6),
           aspect="equal", xlabel="x · column", ylabel="y · row")
    stride = max(1, math.ceil(max(dungeon.width, dungeon.height)/20))
    ax.set_xticks(range(0, dungeon.width, stride))
    ax.set_yticks(range(0, dungeon.height, stride))
    ax.tick_params(length=0, labelsize=7)
    if title:
        ax.set_title(title, loc="left", fontsize=12, pad=12)


def _terrain_legend():
    return [Patch(facecolor=TERRAIN[tile], edgecolor="#ccd3dc", label=label)
            for tile, label in [("#", "Wall"), ("x", "Hazard · blocked"),
                                (".", "Floor · enter 1"), (",", "Gravel · enter 2"),
                                ("~", "Mire · enter 5")]]


def _search_legend(solution=True, frontier_label="Frontier (live)"):
    return [Line2D([], [], marker="s", linestyle="none", color=EXPLORED,
                   markersize=7, label="Expanded state"),
            Line2D([], [], marker="D", linestyle="none", color=FRONTIER,
                   markersize=7, label=frontier_label),
            Line2D([], [], marker="o", linestyle="none", color=GOLD,
                   markeredgecolor=INK, markersize=8, label="Current node"),
            Line2D([], [], color=GOLD, linewidth=3, linestyle="-" if solution else "--",
                   label="Returned route" if solution else "Parent-chain preview")]


def show_map(dungeon, title=None):
    """A fixed map with terrain costs, coordinates, and distinct blocked cells."""
    height = max(5, min(9, dungeon.height / max(dungeon.width, 1) * 11 + 1.4))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, height), layout="constrained")
        _draw_map(ax, dungeon, title=title or dungeon.name)
        ax.legend(handles=_terrain_legend(), loc="upper center", bbox_to_anchor=(.5, -.13),
                  ncol=3, frameon=False, fontsize=9)
        fig.supxlabel(f"S = entrance (underlying cost {_number(dungeon.terrain_cost(*dungeon.start))})  ·  "
                       f"E = exit (enter cost {_number(dungeon.terrain_cost(*dungeon.goal))})  ·  "
                       "four-direction moves; cost is paid on entry", fontsize=9, color="#65758b")
        fig.suptitle("THE CATACOMBS  /  A STATE SPACE TO EXPLORE", fontsize=11,
                     color="#65758b", ha="left", x=.04)
        plt.close(fig)
    return fig


def show_node_path(problem, node, labels=None):
    """Relate a node's actual parent chain to its positions and moves in the maze."""
    nodes = node.path()
    states = [entry.state for entry in nodes]
    labels = list(labels) if labels is not None else [f"node_{i}" for i in range(len(nodes))]
    if len(labels) != len(nodes):
        raise ValueError("Provide one label per node in the parent chain.")
    colours = ["#245b96", "#008596", "#a76b10"]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, max(6.8, len(nodes) * 1.45)), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1), height_ratios=(4, 1))
        maze_ax = fig.add_subplot(grid[0, 0])
        chain_ax = fig.add_subplot(grid[0, 1])
        values_ax = fig.add_subplot(grid[1, :])
        _draw_map(maze_ax, problem.dungeon, title="Positions in the maze")
        # Numbered nodes replace the entrance marker; the exit remains labelled E.
        for label in list(maze_ax.texts):
            if label.get_text() == "S":
                label.remove()
        for start, end in zip(states, states[1:]):
            maze_ax.annotate("", xy=end, xytext=start, zorder=9,
                             arrowprops=dict(arrowstyle="-|>", color=INK, lw=2,
                                             shrinkA=13, shrinkB=13, mutation_scale=13))
        for i, state in enumerate(states):
            colour = colours[i % len(colours)]
            maze_ax.scatter(*state, s=310, color=colour, edgecolor="white", lw=1.5, zorder=10)
            maze_ax.text(*state, str(i), color="white", weight="bold",
                         ha="center", va="center", fontsize=11, zorder=11)
        maze_ax.set_xlabel(
            f"x · column\n0 = entrance S     E = exit {problem.goal}\n"
            f"actions(initial) = {problem.actions(problem.initial)!r}",
            fontsize=10, labelpad=10)

        chain_ax.set(xlim=(0, 1), ylim=(0, 1))
        chain_ax.axis("off")
        chain_ax.set_title("The corresponding Node objects", loc="left", fontsize=12, pad=12)
        centres = np.linspace(.84, .16, len(nodes)) if len(nodes) > 1 else [.5]
        card_height = min(.24, .64 / len(nodes))
        for i, (entry, y) in enumerate(zip(nodes, centres)):
            colour = colours[i % len(colours)]
            chain_ax.add_patch(FancyBboxPatch(
                (.04, y-card_height/2), .92, card_height,
                boxstyle=f"round,pad={card_height*.05},rounding_size={card_height*.1}",
                facecolor="#f5f8fc", edgecolor=colour, linewidth=1.5))
            chain_ax.text(.075, y+card_height*.26, f"{i} · {labels[i]}", color=colour,
                          fontsize=12, weight="bold", va="center")
            chain_ax.text(.075, y-card_height*.04,
                          f"state = {entry.state}    depth = {entry.depth}    path_cost = {entry.path_cost:g}",
                          fontsize=10, va="center")
            parent = "None" if i == 0 else labels[i-1]
            chain_ax.text(.075, y-card_height*.34, f"parent = {parent}    action = {entry.action!r}",
                          fontsize=10, va="center")
            if i:
                top = centres[i-1]-card_height*.6
                bottom = y+card_height*.6
                chain_ax.annotate("", xy=(.5, bottom), xytext=(.5, top),
                                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
                chain_ax.text(.54, (top+bottom)/2, f"child_node(p, {entry.action!r})",
                              fontsize=9, va="center")

        values_ax.axis("off")
        values_ax.text(.01, .83, f"[n.state for n in {labels[-1]}.path()] = {states!r}",
                       fontsize=11, fontfamily="DejaVu Sans Mono")
        values_ax.text(.01, .48, f"{labels[-1]}.solution() = {node.solution()!r}",
                       fontsize=11, fontfamily="DejaVu Sans Mono")
        reached = problem.goal_test(node.state)
        values_ax.text(.01, .12, f"goal_test({node.state}) = {reached}"
                       + ("  ·  exit reached" if reached else "  ·  exit not yet reached"),
                       fontsize=11, color=INK, weight="bold")
        fig.suptitle(f"{problem.dungeon.name} · {node.depth} moves from the entrance",
                     fontsize=17, weight="bold")
        plt.close(fig)
    return fig


def show_maze_atlas(dungeons):
    """A progression of unsearched maps with a shared scale and explicit roles."""
    dungeons = list(dungeons)
    if len(dungeons) != 3:
        raise ValueError("The teaching atlas takes three maps.")
    purposes = ["Branches and loops · DFS / BFS", "Dry bridges or mud · moves / cost",
                "Offset doors · obstacles / heuristic"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), layout="constrained")
        for i, (ax, dungeon, purpose) in enumerate(zip(axes, dungeons, purposes), 1):
            _draw_map(ax, dungeon, title=f"0{i}  {dungeon.name}", annotate=False)
            ax.set_xlim(-.6, 28.6)
            ax.set_ylim(16.6, -.6)
            ax.set_xlabel(purpose, fontsize=10, labelpad=12)
        fig.suptitle("Three architectures, three search questions", fontsize=19, weight="bold")
        fig.legend(handles=_terrain_legend(), loc="outside lower center", ncol=5,
                   frameon=False, fontsize=10)
        plt.close(fig)
    return fig


def show_run(run):
    """Final search evidence; unsuccessful runs never receive a solution route."""
    frame = run.frames[-1] if run.frames else {}
    solved = run.status == "solved"
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(11.5, max(5.8, min(9.5, run.dungeon.height/run.dungeon.width*10+2))),
                               layout="constrained")
        _draw_map(ax, run.dungeon, frame,
                  final_path=run.path if solved else None,
                  title=f"{run.name}  ·  {run.dungeon.name}")
        cost_text = f"Cost {_number(run.cost)}  ·  {_number(run.length)} moves" if solved else "No returned solution"
        fig.suptitle(f"{run.status.upper()}  |  {cost_text}\n"
                     f"{_number(run.expansions)} expansions  ·  {_number(run.unique_expanded)} distinct expanded states  ·  "
                     f"peak {run.memory_label.lower()}: {_number(run.peak_frontier)}",
                     fontsize=10, ha="left", x=.06)
        frontier_label = "Active-path node" if run.memory_label == "Active path nodes" else "Frontier (live)"
        ax.legend(handles=_search_legend(solved, frontier_label), loc="upper center", bbox_to_anchor=(.5, -.13),
                  ncol=2, frameon=False, fontsize=9)
        if not run.frames:
            ax.text(.5, -.29, "Snapshots were disabled: exploration overlays are unavailable.",
                    transform=ax.transAxes, ha="center", color="#67758a", fontsize=9)
        plt.close(fig)
    return fig


def compare_runs(runs):
    """Fixed-geometry side-by-side outcomes, including stopped/failed searches."""
    runs = list(runs.values()) if isinstance(runs, dict) else list(runs)
    if not runs:
        raise ValueError("Provide at least one SearchRun.")
    cols = min(3, len(runs))
    rows = math.ceil(len(runs)/cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.9*rows),
                                 squeeze=False, layout="constrained")
        for ax, run in zip(axes.flat, runs):
            frame = run.frames[-1] if run.frames else {}
            _draw_map(ax, run.dungeon, frame,
                      final_path=run.path if run.status == "solved" else None,
                      title=f"{run.name}  ·  {run.status}", annotate=False)
            cost = _number(run.cost) if run.status == "solved" else "no solution"
            ax.set_xlabel(f"cost {cost}  ·  expansions {_number(run.expansions)}\n"
                          f"peak {run.memory_label.lower()} {_number(run.peak_frontier)}", fontsize=9)
        for ax in list(axes.flat)[len(runs):]:
            ax.set_visible(False)
        fig.suptitle("One problem, different exploration decisions", fontsize=16, weight="bold")
        frontier_label = "Frontier / active path" if any(r.memory_label == "Active path nodes" for r in runs) else "Frontier (live)"
        handles = _search_legend(frontier_label=frontier_label)
        if any(r.status != "solved" for r in runs):
            handles.append(Line2D([], [], color=GOLD, linewidth=3, linestyle="--", label="Parent-chain preview"))
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
        plt.close(fig)
    return fig


def plot_metrics(results):
    """Show method medians plus observed runs. Unsolved cost is unavailable.

    Memory is the recorded algorithm-specific counter, not a byte measurement.
    Multiple maps receive separate rows, avoiding averages across problems.
    """
    maps = list(dict.fromkeys(results["map"].tolist()))
    if not maps:
        raise ValueError("Provide at least one result row.")
    methods = list(dict.fromkeys(results["method"].tolist()))
    colors = ["#218b98", "#ad6f41", "#7467b6", "#497aa8", "#b05071", "#769242", "#3f9291"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(maps), 3, figsize=(14, 3.7*len(maps)),
                                 squeeze=False, layout="constrained")
        for row, name in enumerate(maps):
            group = results[results["map"] == name]
            labels = list(dict.fromkeys(group["method"].astype(str).tolist()))
            for col, (field, title) in enumerate([
                    ("cost", "Returned route cost ↓"), ("expansions", "Expansion events ↓"),
                    ("peak_frontier", "Peak recorded memory counter ↓")]):
                ax = axes[row, col]
                ymax = 1
                for i, method in enumerate(labels):
                    observations = group[group["method"] == method]
                    valid = observations if field != "cost" else observations[observations["status"] == "solved"]
                    values = [float(v) for v in valid[field] if v is not None and np.isfinite(float(v))]
                    if values:
                        ymax = max(ymax, max(values))
                        median = float(np.median(values))
                        ax.bar(i, median, width=.62, color=colors[methods.index(method) % len(colors)],
                               alpha=.87, zorder=3)
                        if len(values) > 1:
                            ax.scatter(i + np.linspace(-.13, .13, len(values)), values,
                                       color=INK, s=10, alpha=.45, zorder=4)
                        label = _number(median)
                        if field == "cost" and len(values) != len(observations):
                            label += f"\n{len(values)}/{len(observations)} solved"
                        ax.annotate(label, (i, max(values)), xytext=(0, 4),
                                    textcoords="offset points", ha="center", fontsize=8)
                    else:
                        status = "/".join(dict.fromkeys(observations["status"].astype(str)))
                        ax.text(i, .025, status, transform=ax.get_xaxis_transform(),
                                rotation=90, ha="center", va="bottom", color="#8a4e60", fontsize=8)
                ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
                ax.set_title(f"{name}\n{title}", loc="left", fontsize=11)
                ax.set_ylim(0, ymax*1.2)
                ax.set_xlim(-.6, len(labels)-.4)
                ax.grid(axis="y", alpha=.6, zorder=0)
        fig.suptitle("Compare quality, work, and memory on the same map", fontsize=16, weight="bold")
        fig.supxlabel("Bars: median per method; dots: repeated observations. Missing cost means no returned solution.\n"
                       "Memory: queue/heap entries for frontier methods; active-path nodes for depth-first recursion.\n"
                       "This counter excludes explored sets, best-cost tables, and recorded animation frames.",
                       fontsize=9, color="#65758b")
        plt.close(fig)
    return fig


def show_heuristic(dungeon, h, truth=None):
    """Compare h(state) to supplied exact remaining costs (mapping or callable).

    ``truth`` is optional; this function never computes shortest paths.
    Cells without a finite value are masked, including unreachable cells.
    """
    def value(source, state):
        return source(state) if callable(source) else source.get(state, float("inf"))

    sources = [("h(n) · estimated remaining cost", h)]
    if truth is not None:
        sources.append(("h*(n) · exact remaining cost", truth))
    arrays = []
    for _, source in sources:
        a = np.full((dungeon.height, dungeon.width), np.nan)
        for y in range(dungeon.height):
            for x in range(dungeon.width):
                if dungeon.passable(x, y):
                    v = float(value(source, (x, y)))
                    if np.isfinite(v):
                        a[y, x] = v
        arrays.append(a)
    maximum = max((np.nanmax(a) for a in arrays if np.isfinite(a).any()), default=1)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(sources), figsize=(6.6*len(sources), max(4.5, min(8, dungeon.height/dungeon.width*6.3+1.5))),
                                 squeeze=False, layout="constrained")
        cmap = plt.get_cmap("YlGnBu").copy()
        cmap.set_bad(TERRAIN["#"])
        for ax, (title, _), a in zip(axes.flat, sources, arrays):
            im = ax.imshow(np.ma.masked_invalid(a), cmap=cmap, vmin=0, vmax=max(1, maximum))
            if dungeon.width <= 25 and dungeon.height <= 20:
                for y, x in zip(*np.where(np.isfinite(a))):
                    ax.text(x, y, _number(a[y, x]), ha="center", va="center", fontsize=6,
                            color="white" if a[y, x] > maximum*.55 else INK)
            for point, label in ((dungeon.start, "S"), (dungeon.goal, "E")):
                ax.annotate(label, point, xytext=(0, 0), textcoords="offset points", ha="center", va="center",
                            color=INK, weight="bold", fontsize=8,
                            bbox=dict(facecolor=GOLD, edgecolor="white", boxstyle="round,pad=.15"))
            ax.set(title=title, xlabel="x · column", ylabel="y · row")
            stride = max(1, math.ceil(max(dungeon.width, dungeon.height)/18))
            ax.set_xticks(range(0, dungeon.width, stride))
            ax.set_yticks(range(0, dungeon.height, stride))
            ax.tick_params(labelsize=7, length=0)
        fig.colorbar(im, ax=list(axes.flat), label="Remaining cost · same scale in every panel" if truth is not None
                     else "Estimated remaining cost", shrink=.68)
        fig.suptitle("An estimate sees proximity; exact distance also sees the obstacles" if truth is not None
                     else "A heuristic estimates the cost still ahead", fontsize=14, weight="bold")
        if truth is not None:
            valid = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
            over = np.count_nonzero(arrays[0][valid] > arrays[1][valid] + 1e-9)
            fig.supxlabel(f"{over} finite reachable states overestimate the supplied exact cost. "
                           "Dark cells: blocked or no finite distance.", fontsize=9)
        plt.close(fig)
    return fig


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _fallback_svg(dungeon):
    unit = 22
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dungeon.width*unit} {dungeon.height*unit}" '
             'role="img" aria-label="Static dungeon map: S entrance, E exit">']
    for y, row in enumerate(dungeon.rows):
        for x, tile in enumerate(row):
            color = TERRAIN.get(tile, TERRAIN["."])
            parts.append(f'<rect x="{x*unit}" y="{y*unit}" width="22" height="22" fill="{color}" stroke="#758497" stroke-width=".3"/>')
            if (x, y) in (dungeon.start, dungeon.goal):
                label = "S" if (x, y) == dungeon.start else "E"
                parts.append(f'<text x="{x*unit+11}" y="{y*unit+15}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="{INK}">{label}</text>')
    return "".join(parts) + "</svg>"


def _player_document(run):
    if not run.frames:
        raise ValueError("Rerun the search with record=True to capture animation frames.")
    dungeon = run.dungeon
    terrain = []
    for y, row in enumerate(dungeon.rows):
        terrain.append([dict(tile=tile, name=_terrain_name(tile), passable=bool(dungeon.passable(x, y)),
                             cost=dungeon.terrain_cost(x, y) if dungeon.passable(x, y) else None)
                        for x, tile in enumerate(row)])
    payload = dict(name=run.name, map_name=dungeon.name, rows=dungeon.rows,
                   width=dungeon.width, height=dungeon.height, terrain=terrain,
                   start=dungeon.start, goal=dungeon.goal, frames=run.frames,
                   status=run.status, cost=run.cost, length=run.length,
                   expansions=run.expansions, generated=run.generated,
                   unique_expanded=run.unique_expanded, peak_frontier=run.peak_frontier,
                   memory_label=run.memory_label, seconds=run.seconds,
                   solution=run.path if run.status == "solved" else [], settings=run.settings)
    assets = Path(__file__).with_name("assets")
    encoded = json.dumps(_json_safe(payload), separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return ((assets / "player.html").read_text(encoding="utf-8")
            .replace("__STYLE__", (assets / "player.css").read_text(encoding="utf-8"))
            .replace("__DATA__", encoded)
            .replace("__SCRIPT__", (assets / "player.js").read_text(encoding="utf-8"))
            .replace("__FALLBACK__", _fallback_svg(dungeon)))


def animate_run(run):
    """Return an offline HTML player: no widget kernel, server, CDN, or FFmpeg."""
    document = _player_document(run)
    return IFrame("about:blank", width="100%", height=1020, extras=[
        'title="Catacombs search replay"', 'sandbox="allow-scripts"',
        'style="border:0;background:#f3f6fa;border-radius:16px"',
        'srcdoc="' + html.escape(document, quote=True) + '"',
    ])


def save_player(run, filename):
    """Write the same interactive replay to one portable HTML file."""
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_player_document(run), encoding="utf-8")
    return str(path.resolve())


def _graph_layout(problem):
    """Columns by fewest edges from the initial state; states spread vertically."""
    from collections import deque
    layer = {problem.initial: 0}
    queue = deque([problem.initial])
    while queue:
        state = queue.popleft()
        for child in problem.edges.get(state, {}):
            if child not in layer:
                layer[child] = layer[state] + 1
                queue.append(child)
    for state in problem.states():
        layer.setdefault(state, max(layer.values()) + 1)
    columns = {}
    for state in problem.states():
        columns.setdefault(layer[state], []).append(state)
    return {state: (depth, (len(states) - 1) / 2 - i)
            for depth, states in columns.items() for i, state in enumerate(states)}


def show_graph(problem, positions=None, title=None):
    """Draw a GraphProblem: states as circles, ACTION-COST on every edge, h under a state when stored.

    The drawing reads ``problem.edges`` only: it makes no ``actions`` or
    ``goal_test`` call, so the recorded traces stay empty.
    """
    positions = dict(positions or getattr(problem, "positions", None) or _graph_layout(problem))
    missing = [s for s in problem.states() if s not in positions]
    if missing:
        raise ValueError(f"Provide a position for every state; missing {missing}")
    edges = problem.edges
    goals = problem.goal if isinstance(problem.goal, list) else [problem.goal]
    directed = any(edges.get(v, {}).get(u) != cost
                   for u, neighbours in edges.items() for v, cost in neighbours.items())
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    span_x, span_y = (max(xs) - min(xs)) or 1, (max(ys) - min(ys)) or 1
    radius_pt = 22
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(2.6 * span_x + 2.5, 1.7 * span_y + 1.9), layout="constrained")
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(min(xs) - .55, max(xs) + .55)
        ax.set_ylim(min(ys) - .55, max(ys) + .55)
        drawn = set()
        for u, neighbours in edges.items():
            for v, cost in neighbours.items():
                if (v, u) in drawn:
                    continue
                two_way = edges.get(v, {}).get(u) == cost
                drawn.add((u, v))
                (x0, y0), (x1, y1) = positions[u], positions[v]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                            arrowprops=dict(arrowstyle="-" if two_way else "-|>", color=INK, lw=1.6,
                                            shrinkA=radius_pt, shrinkB=radius_pt, mutation_scale=14))
                dx, dy = x1 - x0, y1 - y0
                norm = math.hypot(dx, dy) or 1
                ax.annotate(_number(cost), xy=((x0 + x1) / 2, (y0 + y1) / 2), zorder=5,
                            xytext=(-dy / norm * 11, dx / norm * 11), textcoords="offset points",
                            ha="center", va="center", fontsize=10, weight="bold", color="#a76b10",
                            bbox=dict(boxstyle="round,pad=.15", facecolor="white", edgecolor="none", alpha=.9))
        for state, (x, y) in positions.items():
            face = GOLD if state == problem.initial else "#cfe8d8" if state in goals else "white"
            ax.scatter(x, y, s=(2 * radius_pt) ** 2, facecolor=face, edgecolor=INK, linewidth=1.6, zorder=3)
            if problem.heuristic:   # name above, h below, both inside the circle
                ax.annotate(str(state), (x, y), xytext=(0, 5), textcoords="offset points",
                            ha="center", va="center", fontsize=11, weight="bold", zorder=4)
                ax.annotate(f"h={_number(problem.h(state))}", (x, y), xytext=(0, -8), textcoords="offset points",
                            ha="center", va="center", fontsize=8, color="#65758b", zorder=4)
            else:
                ax.text(x, y, str(state), ha="center", va="center", fontsize=11, weight="bold", zorder=4)
        ax.legend(handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=GOLD, markeredgecolor=INK,
                   markersize=10, label=f"Initial state {problem.initial}"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="#cfe8d8", markeredgecolor=INK,
                   markersize=10, label=f"Goal {problem.goal}"),
            Line2D([], [], color="#a76b10", linewidth=0, marker="$5$", markersize=9,
                   label="ACTION-COST of the edge")],
            loc="upper center", bbox_to_anchor=(.5, -.02), ncol=3, frameon=False, fontsize=9)
        kind = "directed" if directed else "undirected"
        ax.set_title(title or f"{kind.capitalize()} graph · start {problem.initial}, goal {problem.goal}"
                     + (" · h stored in each state" if problem.heuristic else ""), fontsize=12)
        plt.close(fig)
    return fig
