"""The original torch-lit dungeon renderer, driven by the notebook's real traces.

Rendering is passive: it never runs a solver, imports solver code, or writes any
progress file. The hero waits while the algorithm searches, then walks only a returned
solution. Movie encoding happens outside all search timings.
"""
from __future__ import annotations

import base64
import html
import os
from pathlib import Path
import subprocess
import tempfile
import math

from IPython.display import HTML
from .core import _sample_frames
from .models import DungeonPathProblem

_CURRENT_PATH = (238, 244, 255)
_BACKTRACK = (255, 164, 87)


def _waiting_cells(run, frame):
    """Actual container order, or the deferred actions of suspended DLS calls.

    Projection uses only the maze's known move vectors and recorded actions.
    No successor requests or search are performed by the movie. Recursion cycle
    checks use each suspended call's own parent chain, not a global visited set.
    Multiple pending routes to one cell share the earliest visible cell rank.
    """
    recursive = (frame.get('frontier_kind') == 'path'
                 or run.memory_label == 'Active path nodes')
    states = []
    if recursive:
        if frame['event'] == 'limit' and frame.get('current') is not None:
            return [tuple(frame['current'])]  # Root is about to enter this pass.
        for call in frame.get('deferred_actions') or []:
            ancestors = {tuple(state) for state in call['path']}
            x, y = call['parent']
            for action in call['actions']:
                dx, dy = DungeonPathProblem.DIRECTIONS[action]
                state = (x+dx, y+dy)
                if state not in ancestors and run.dungeon.passable(*state):
                    states.append(state)
    else:
        states = [tuple(node['state']) for node in frame['frontier'] if not node['stale']]
    return list(dict.fromkeys(states))


def _video(data, caption, width=1100):
    encoded = base64.b64encode(data).decode('ascii')
    return HTML(
        '<figure style="margin:12px 0">'
        f'<video controls loop playsinline preload="metadata" width="{int(width)}" '
        'style="max-width:100%;height:auto;border-radius:12px;background:#0b0a12" '
        'aria-label="Dungeon exploration, route reveal and hero walk">'
        f'<source src="data:video/mp4;base64,{encoded}" type="video/mp4">'
        'Your browser cannot play this MP4. Download the MP4 and open it in a video player.'
        '</video><figcaption style="font:14px/1.6 sans-serif;color:#55657b">'
        + html.escape(caption) + '</figcaption></figure>')


def original_animation():
    """Display the original notebook movie, preserved byte for byte."""
    return _video((Path(__file__).with_name('assets')/'original_gallery.mp4').read_bytes(),
                  'Original introduction: exploration → route → walking. '
                  'Use the exercise results for current measurements.')


class _RecordedPlayback:
    """A small adapter for SceneRenderer, using actual frontier snapshots."""
    def __init__(self, run):
        self.run = run
        self.path_states = run.path if run.status == 'solved' else []
        self.phase = 'explore'
        self.path_j = 0.0
        self.walk_t = 0.0
        self.apply(run.frames[0])

    def apply(self, frame):
        self.frame = frame
        self.is_path = (frame.get('frontier_kind') == 'path'
                        or self.run.memory_label == 'Active path nodes')
        explored = frame.get('pass_explored', frame['explored'])
        self.expanded = {tuple(s) for s in explored}
        self.waiting = _waiting_cells(self.run, frame)

    @property
    def frontier(self):
        # Once search has stopped, pending alternatives do not compete with the route.
        return set(self.waiting) if self.phase == 'explore' else set()

    @property
    def frontier_ranks(self):
        """Persistent ranks of currently waiting cells, for every approach.

        The gold current node has already been selected. Its expansion can put
        new children ahead of rank 1. Ranks describe this snapshot, even if a
        later movie snapshot skips over intervening frontier updates.
        """
        if self.phase != 'explore':
            return []
        return list(enumerate(self.waiting, 1))

    def hero_pos(self):
        if self.phase not in ('walk', 'done') or not self.path_states:
            return tuple(map(float, self.run.dungeon.start))
        j = min(int(self.walk_t), len(self.path_states)-1)
        a = self.path_states[j]
        b = self.path_states[min(j+1, len(self.path_states)-1)]
        fraction = self.walk_t-j
        return (a[0]+fraction*(b[0]-a[0]), a[1]+fraction*(b[1]-a[1]))

    def recent(self, n=6):
        current = self.frame.get('current')
        if (self.phase == 'explore' and current is not None
                and not self.frame.get('stale_current') and self.frame['event'] != 'limit'):
            return [(tuple(current), 0.0)]
        return []

    def visible_path(self):
        if self.phase == 'explore':
            return []
        return self.path_states[:int(self.path_j)]

    @property
    def exploration_path(self):
        """The selected node's recorded parent chain, separate from the solution.

        Successive selections may belong to different branches. Never connect
        their coordinates into a fictional walk, or use the final answer to
        fill a missing trace. Discarded entries and stopped searches have no
        active exploration branch to display.
        """
        if (self.phase != 'explore' or self.frame.get('current') is None
                or _event(self.frame) in ('limit', 'pass_end', 'skip', 'failure', 'budget')):
            return []
        return [tuple(state) for state in self.frame.get('path', [])]

    @property
    def return_edge(self):
        """One explicitly recorded recursive return, never inferred across a gap."""
        source = self.frame.get('return_from')
        target = self.frame.get('current')
        if (self.phase == 'explore' and _event(self.frame) == 'backtrack'
                and source is not None and target is not None):
            return tuple(source), tuple(target)
        return None


def _search_snapshots(run, count):
    """Pair displayed frontier updates with their real following selection.

    Sampling may add that selection and intervening stale-entry skips. Gaps in
    the source trace are never filled with invented states. Display annotations
    live on shallow copies, leaving the recorded search untouched.
    """
    frames = run.frames
    indexes = {id(frame): i for i, frame in enumerate(frames)}
    chosen = (set(range(len(frames))) if count is None else
              {indexes[id(frame)] for frame in _sample_frames(frames, count)})
    paired = set()
    for i in sorted(chosen):
        frame = frames[i]
        recursive = (frame.get('frontier_kind') == 'path'
                     or run.memory_label == 'Active path nodes')
        if _strategy(run, frame) == 'random_walk':
            # A random walk keeps no frontier: every recorded move is shown as it is.
            if _event(frame) not in ('start', 'select', 'goal'):
                chosen.discard(i)
            continue
        eligible = ('limit', 'expand', 'cutoff', 'backtrack') if recursive else ('start', 'expand')
        if _event(frame) not in eligible or not _waiting_cells(run, frame):
            continue
        previous_step = frame['step']
        for j in range(i+1, len(frames)):
            following = frames[j]
            if following['step'] != previous_step+1:
                break
            previous_step = following['step']
            event = _event(following)
            if event == 'skip' or (recursive and event == 'backtrack'):
                continue
            if event in ('select', 'goal'):
                if tuple(following['current']) == _waiting_cells(run, frame)[0]:
                    chosen.update(range(i+1, j+1))
                    paired.add(i)
            break
    previous_step = None
    for i in sorted(chosen):
        frame = frames[i]
        yield dict(frame, _next_selection_shown=i in paired,
                   _skipped_events=0 if previous_step is None else
                   max(0, frame['step']-previous_step-1))
        previous_step = frame['step']


def _timeline(run, explore_frames=96, path_frames=22, walk_frames=60, hold=18,
              *, fps=15, step_seconds=0.4):
    """Hold real events long enough to read; retain IDS restarts when sampling.

    explore_frames sets the initial sample size; matching selections may be
    added to preserve frontier-order transitions. Each snapshot is held for
    step_seconds; restarts and terminal events receive a longer pause.
    No intermediate states, queue membership or work counts are invented.
    """
    if not run.frames:
        raise ValueError('Rerun with record=True before animating.')
    for frame in _search_snapshots(run, explore_frames):
        pause = step_seconds
        if frame['event'] in ('limit', 'pass_end', 'backtrack', 'goal', 'failure', 'budget', 'cutoff'):
            pause = max(pause, 0.7)
        if _strategy(run, frame) == 'ids' and frame['event'] in ('limit', 'pass_end'):
            pause = max(pause, 1.2)
        for _ in range(max(1, round(fps*pause))):
            yield ('explore', frame, 0.0, 0.0)
    if run.status == 'solved':
        for k in range(1, path_frames+1):
            yield ('path', run.frames[-1], len(run.path)*k/path_frames, 0.0)
        for k in range(1, walk_frames+1):
            yield ('walk', run.frames[-1], float(len(run.path)), (len(run.path)-1)*k/walk_frames)
    for _ in range(hold):
        yield ('done', run.frames[-1], float(len(run.path)) if run.status == 'solved' else 0.0,
               float(len(run.path)-1) if run.status == 'solved' else 0.0)


def _render_setup(run):
    os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    try:
        import pygame
    except ImportError as exc:
        raise ImportError(
            'The dungeon renderer needs pygame in this notebook kernel. '
            'Run the setup cell, or run %pip install "pygame-ce>=2.5.8,<3" '
            'in a notebook cell, then retry the animation.'
        ) from exc
    from engine.render.renderer import SceneRenderer
    from engine.render import theme, tiles

    pygame.font.init()  # No game window, sound device, application or progress file.

    renderer = SceneRenderer((1280, 720))
    renderer.prepare(run.dungeon)
    # Reuse the original textures/light/hero, with the maze centred in one view.
    # The renderer reserves room for a side panel; this lab has none.
    renderer.ts = max(8, min(56, int(min(1190/run.dungeon.width, 490/run.dungeon.height))))
    renderer.ox = (1280-run.dungeon.width*renderer.ts)//2
    renderer.oy = 112+(490-run.dungeon.height*renderer.ts)//2
    renderer.tileset = tiles.build_tileset(renderer.ts)
    renderer._build_static()
    return pygame, theme, renderer, pygame.Surface(renderer.size)


def _strategy(run, frame):
    """Use the solver identity, never a user's arbitrary display label, for rules."""
    known = {
        'depth_first_search': 'dfs', 'breadth_first_search': 'bfs',
        'depth_limited_search': 'dls', 'iterative_deepening_search': 'ids',
        'uniform_cost_search': 'ucs', 'greedy_best_first_search': 'greedy',
        'a_star_search': 'astar',
        'random_walk_navigation': 'random_walk',
    }
    strategy = known.get(run.settings.get('solver'))
    if strategy:
        return strategy
    # Older/custom runs can still show the actual container discipline.
    return {'stack': 'dfs', 'queue': 'bfs', 'priority': 'priority',
            'path': 'dls'}.get(frame.get('frontier_kind'), 'custom')


def _event(frame):
    if frame.get('stale_current') or frame.get('note', '').lower().startswith('skip stale'):
        return 'skip'
    if frame.get('note', '').lower().startswith('initialize'):
        return 'start'
    return frame['event']


def _explanation(run, viz):
    frame, phase = viz.frame, viz.phase
    if _strategy(run, frame) == 'random_walk':
        if phase == 'path':
            return 'ROUTE  |  The recorded walk without its detours: the route to the exit, each cell at most once.'
        if phase == 'walk':
            return 'WALK  |  The hero follows that route. No move is undone.'
        if phase == 'done' and run.status == 'solved':
            return (f'COMPLETE  |  {run.expansions} simulated moves  |  '
                    f'route of {run.length} moves, cost {run.cost:g}.')
        if phase == 'explore':
            if _event(frame) == 'start':
                return 'SIMULATE  |  Uniform random walk from the entrance; revisits allowed.'
            if _event(frame) == 'goal':
                return 'GOAL REACHED  |  End the simulation; keep the walk without its detours.'
            if _event(frame) == 'select':
                return f'MOVE {frame["expansions"]}  |  {tuple(frame["current"])} reached by a uniformly sampled legal action.'
    if phase == 'path':
        return 'ROUTE  |  Search has finished. Reveal the returned sequence of moves.'
    if phase == 'walk':
        return 'WALK  |  The hero executes the route. No more search expansions occur.'
    event = _event(frame)
    if phase == 'done':
        event = run.status
    if event == 'solved':
        return f'COMPLETE  |  {run.length} moves  |  cost {run.cost:g}  |  {run.expansions:,} expansion calls.'
    if event == 'budget':
        return f'WORK LIMIT  |  {frame["expansions"]:,} expansions used. No route returned; reachability remains unresolved.'
    if event == 'failure':
        return 'FAILURE  |  Search exhausted its possibilities without finding a route.'
    if event == 'limit':
        if 'pass_explored' not in frame:
            return f'RESTART  |  DLS limit {frame["depth_limit"]}. Older cumulative trace: rerun the search for per-pass colours.'
        return f'BEGIN PASS  |  Limit {frame["depth_limit"]}. Start again at the entrance; the active branch is empty.'
    if event == 'pass_end':
        reason = ('some branches reached the depth bound' if frame.get('return_status') == 'cutoff'
                  else 'all branches exhausted without reaching the goal')
        return f'PASS ENDED  |  Limit {frame["depth_limit"]}: {reason}. The frontier is empty.'
    if event == 'backtrack':
        reason = 'child returned CUTOFF' if frame.get('return_status') == 'cutoff' else 'child returned failure'
        return (f'RETURN {tuple(frame["return_from"])} → {tuple(frame["current"])}  |  '
                f'{reason}; resume the parent\'s successor loop.')
    if event == 'cutoff':
        if phase == 'done' or frame.get('current') is None:
            return 'DEPTH LIMIT  |  No route returned within the allowed depth. A deeper route may still exist.'
        return f'DEPTH LIMIT  |  Depth {frame["depth"]} reaches limit {frame["depth_limit"]}: stop this branch and backtrack.'
    if event == 'goal':
        return 'GOAL SELECTED  |  A solution is found. The hero still waits while the route is recovered.'
    if event == 'skip':
        return 'SKIP  |  This queue entry has an outdated cost. It is discarded without expansion.'
    if event == 'start':
        return 'START  |  Put the entrance into the frontier. The hero waits while the algorithm searches.'
    current = tuple(frame['current']) if frame.get('current') is not None else '—'
    if event == 'expand':
        if viz.frontier_ranks:
            n = len(viz.waiting)
            next_step = ('Rank 1 is selected next.' if frame.get('_next_selection_shown')
                         else 'Ranks describe this instant; the next selection may be outside this sample.')
            label = 'CHILDREN WAITING' if viz.is_path else 'FRONTIER UPDATED'
            return f'{label}  |  {n} waiting cells, ranked 1–{n}. {next_step}'
        if viz.is_path:
            return f'EXPAND {current}  |  Successors processed along the active recursion branch.'
        if not viz.waiting:
            return f'EXPAND {current}  |  Successors processed. No waiting node remains.'
        return f'EXPAND {current}  |  Successors processed; frontier updated. The next selection is outside this sample.'
    strategy = _strategy(run, frame)
    g, h, f = frame['g'], frame['h'], frame['f']
    reasons = {
        'dfs': f'last inserted into the stack; depth {frame["depth"]}',
        'bfs': f'oldest entry in the queue; depth {frame["depth"]}',
        'ucs': f'smallest current path cost g = {g:g}',
        'greedy': f'smallest estimate h = {h:.2f}; g does not set the priority',
        'astar': f'smallest f = g + h: {g:g} + {h:.2f} = {f:.2f}',
        'priority': 'next entry selected by the recorded priority queue',
        'custom': 'next node selected by your solver',
        'dls': f'depth {frame["depth"]} / limit {frame["depth_limit"]}; test the goal before the bound',
        'ids': f'depth {frame["depth"]} / limit {frame["depth_limit"]}; test the goal before the bound',
    }
    verb = 'ENTER' if viz.is_path else 'SELECT'
    return f'{verb} {current}  |  {reasons[strategy]}. Cyan ranks = still waiting.'


def _text(surface, theme, text, xy, *, size=18, color=None, width=1200, bold=False, ui=True):
    """Fit long maze/method labels without overlaying adjacent information."""
    size = int(size)
    font = theme.font(size, bold=bold, ui=ui)
    while size > 12 and font.size(text)[0] > width:
        size -= 1
        font = theme.font(size, bold=bold, ui=ui)
    if font.size(text)[0] > width:
        while text and font.size(text+'…')[0] > width:
            text = text[:-1]
        text += '…'
    surface.blit(font.render(text, True, color or theme.PARCHMENT), xy)


def _draw_costs(run, renderer, surface, pygame, theme):
    """Small corner badges distinguish entry costs c from central frontier ranks."""
    for state in run.dungeon.floor_positions():
        value = run.dungeon.terrain_cost(*state)
        label = f'c{value:.3g}'
        size = min(14, max(7, renderer.ts//3))
        font = theme.font(size, bold=True, ui=True)
        while size > 5 and font.size(label)[0] > renderer.ts-5:
            size -= 1
            font = theme.font(size, bold=True, ui=True)
        glyph = font.render(label, True, (245, 231, 200))
        cell = renderer.cell_rect(*state)
        box = glyph.get_rect(topleft=(cell.x+2, cell.y+1)).inflate(2, 0)
        pygame.draw.rect(surface, theme.BG_DEEP, box, border_radius=2)
        surface.blit(glyph, (cell.x+2, cell.y+1))


def _draw_frame(run, viz, renderer, surface, pygame, theme, t, *, show_costs=None):
    renderer.draw(surface, viz, t, orb=False)
    if show_costs is None:
        show_costs = hasattr(run.dungeon, 'entry_costs')
    if show_costs:
        _draw_costs(run, renderer, surface, pygame, theme)
    frame = viz.frame
    if viz.phase == 'explore':
        # One actual root-to-current route for every approach. The thin white
        # thread updates with the selected node; only the final route is gold.
        points = [renderer.center(*state) for state in viz.exploration_path]
        if len(points) > 1:
            width = max(2, renderer.ts//14)
            pygame.draw.lines(surface, theme.BG_DEEP, False, points, width+3)
            pygame.draw.lines(surface, _CURRENT_PATH, False, points, width)
            for point in points[1:-1]:
                pygame.draw.circle(surface, _CURRENT_PATH, point, max(1, width))
            # Restore the stationary hero above the thread at the entrance.
            renderer._draw_hero(surface, viz.hero_pos(), t)
        if viz.return_edge:
            source, target = viz.return_edge
            a, b = renderer.center(*source), renderer.center(*target)
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            ux, uy = dx/length, dy/length
            # Stop before the parent's outline; show only this real return edge.
            tip = (b[0]-ux*renderer.ts*.3, b[1]-uy*renderer.ts*.3)
            pygame.draw.line(surface, _BACKTRACK, a, tip, 3)
            wing = max(4, renderer.ts*.12)
            pygame.draw.polygon(surface, _BACKTRACK, [tip,
                (tip[0]-ux*wing-uy*wing, tip[1]-uy*wing+ux*wing),
                (tip[0]-ux*wing+uy*wing, tip[1]-uy*wing-ux*wing)])
        for rank, state in viz.frontier_ranks:
            center = renderer.center(*state)
            label = str(rank)
            size = max(8, min(19, renderer.ts//2))
            font = theme.font(size, bold=True, ui=True)
            while size > 5 and font.size(label)[0] > renderer.ts-6:
                size -= 1
                font = theme.font(size, bold=True, ui=True)
            glyph = font.render(label, True, theme.CYAN)
            box = glyph.get_rect(center=center).inflate(4, 2)
            pygame.draw.rect(surface, theme.BG_DEEP, box, border_radius=2)
            surface.blit(glyph, glyph.get_rect(center=center))
        current = frame.get('current')
        if current is not None and frame['event'] != 'limit':
            rect = renderer.cell_rect(*current).inflate(-2, -2)
            color = (theme.DANGER if frame.get('stale_current') else
                     _BACKTRACK if _event(frame) == 'cutoff' else theme.GOLD)
            pygame.draw.rect(surface, theme.PARCHMENT, rect, width=1, border_radius=4)
            pygame.draw.rect(surface, color, rect.inflate(-4, -4), width=3, border_radius=3)

    pygame.draw.rect(surface, (18, 16, 29), (0, 0, 1280, 94))
    pygame.draw.line(surface, (77, 68, 96), (28, 94), (1252, 94))
    _text(surface, theme, run.dungeon.name, (28, 12), size=27, color=theme.GOLD, width=820, ui=False)
    active = {'explore': 0, 'path': 1, 'walk': 2, 'done': 2 if run.status == 'solved' else 0}[viz.phase]
    random_walk = _strategy(run, frame) == 'random_walk'
    phases = ('1 SIMULATE', '2 ROUTE', '3 WALK') if random_walk else ('1 SEARCH', '2 ROUTE', '3 WALK')
    for i, label in enumerate(phases):
        x = 884+i*124
        color = theme.GOLD if i == active else (141, 135, 153)
        _text(surface, theme, label, (x, 21), size=16, color=color, bold=i == active, width=115)
        if i == active:
            pygame.draw.line(surface, color, (x, 46), (x+100, 46), 2)
    rules = {'dfs': 'stack · LIFO', 'bfs': 'queue · FIFO', 'ucs': 'priority = g',
             'greedy': 'priority = h', 'astar': 'priority = g + h',
             'dls': 'depth-first · bounded', 'ids': 'repeat DLS · deeper limits',
             'priority': 'priority queue', 'custom': 'recorded search',
             'random_walk': f'uniform legal moves · seed {run.settings.get("seed", 7)}'}
    rule = rules[_strategy(run, frame)]
    if frame.get('depth_limit') is not None:
        depth = '—' if frame.get('current') is None or _event(frame) == 'limit' else frame['depth']
        rule = f'depth {depth} / limit {frame["depth_limit"]}  |  white = path to the current node'
    elif show_costs:
        rule += '  |  c = entry cost ('+getattr(run.dungeon, 'cost_label', 'terrain')+')'
    _text(surface, theme, run.name+'  |  '+rule, (28, 58),
          size=19, color=theme.PARCHMENT, width=760)
    if random_walk:
        visited_cells = {tuple(s) for s in frame['explored']}
        if frame.get('current') is not None:
            visited_cells.add(tuple(frame['current']))
        count = f'{frame["expansions"]} simulated moves  |  {len(visited_cells)} visited cells'
    elif viz.phase != 'explore' and run.status == 'solved':
        count = f'{run.length} moves  |  cost {run.cost:g}  |  {run.expansions:,} expansions'
    elif frame.get('depth_limit') is not None and 'pass_expansions' in frame:
        count = f'expansions: this pass {frame["pass_expansions"]:,}  |  total {frame["expansions"]:,}'
    elif viz.is_path:
        count = f'{frame["expansions"]:,} cumulative expansions'
    else:
        count = f'{frame["expansions"]:,} expansions  |  {len(viz.waiting)} waiting cells'
    _text(surface, theme, count, (824, 59), size=16, width=428)

    pygame.draw.rect(surface, (18, 16, 29), (0, 624, 1280, 96))
    pygame.draw.line(surface, (77, 68, 96), (28, 624), (1252, 624))
    _text(surface, theme, _explanation(run, viz), (28, 638), size=19, width=1224)
    visited = 'explored this pass' if viz.is_path and 'pass_explored' in frame else 'explored'
    legend = [(theme.GOLD, 'current / final route'), (_CURRENT_PATH, 'path to current node')]
    if random_walk:
        visited = 'visited cells'
        legend = [(theme.GOLD, 'current cell / final route'), (_CURRENT_PATH, 'walk without detours')]
    else:
        legend.append((theme.CYAN, 'waiting ranks · now'))
    if viz.is_path:
        legend.append((_BACKTRACK, 'depth bound / return'))
    legend.append((theme.VIOLET, visited))
    x = 28
    spacing = 912//len(legend)
    for color, label in legend:
        pygame.draw.rect(surface, color, (x, 682, 11, 11), border_radius=2)
        _text(surface, theme, label, (x+19, 677), size=14, color=(207, 201, 220), width=spacing-21)
        x += spacing
    skipped = frame.get('_skipped_events', 0)
    position = f'trace step {frame["step"]:,}'
    if skipped:
        position += f' · {skipped:,} events skipped'
    _text(surface, theme, position, (944, 678), size=13,
          color=theme.GOLD_SOFT if skipped else (165, 158, 180), width=308)


def save_movie(run, filename, fps=15, explore_frames=96, path_frames=22,
               walk_frames=60, hold=18, step_seconds=0.4, show_costs=None):
    """Encode an MP4 using the original renderer and a bundled FFmpeg binary."""
    if not math.isfinite(fps) or fps <= 0 or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in (path_frames, walk_frames, hold)):
        raise ValueError('fps and all phase frame counts must be positive.')
    if explore_frames is not None and (not isinstance(explore_frames, int)
            or isinstance(explore_frames, bool) or explore_frames < 1):
        raise ValueError('explore_frames must be positive, or None for every recorded event.')
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError('step_seconds must be finite and positive.')
    if not run.frames:
        raise ValueError('Rerun with record=True before animating.')
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        if exc.name != 'imageio_ffmpeg':
            raise
        raise ModuleNotFoundError(
            'The animation needs imageio-ffmpeg in this notebook kernel. '
            'Run the setup cell, or run %pip install "imageio-ffmpeg>=0.6,<1" '
            'in a notebook cell, then retry the animation.',
            name='imageio_ffmpeg',
        ) from exc
    pygame, theme, renderer, surface = _render_setup(run)
    viz = _RecordedPlayback(run)
    path = Path(filename).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
               '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '1280x720',
               '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '25',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for i, (phase, frame, reveal, walk) in enumerate(_timeline(
                    run, explore_frames, path_frames, walk_frames, hold,
                    fps=fps, step_seconds=step_seconds)):
                viz.apply(frame)
                viz.phase, viz.path_j, viz.walk_t = phase, reveal, walk
                _draw_frame(run, viz, renderer, surface, pygame, theme, 2+i/fps, show_costs=show_costs)
                process.stdin.write(pygame.image.tobytes(surface, 'RGB'))
            process.stdin.close()
            process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise
        if process.returncode:
            errors.seek(0)
            raise RuntimeError('Movie encoding failed: '+errors.read().decode(errors='replace'))
    return str(path)


def _movie_caption(run, *, show_costs=None):
    """Explain only the visual conventions relevant to this recorded run."""
    parts = [f'{run.name} · {run.dungeon.name}. Search → route → walking.',
             'White traces the path from the entrance to the current node; '
             'gold shows the returned route.',
             'Cyan numbers rank waiting cells at the displayed step; '
             'new children can change their order.']
    if run.memory_label == 'Active path nodes':
        parts.append('For this recursive traversal, ranks list deferred alternatives, '
                     'deepest call first. Orange marks a depth cutoff '
                     'or a recorded return to the parent.')
    if show_costs is None:
        show_costs = hasattr(run.dungeon, 'entry_costs')
    if show_costs:
        parts.append('Corner labels such as c7 give the fixed entry cost of a cell.')
    parts.append('Gaps in sampled traces are marked.')
    return ' '.join(parts)


def animate_run(run, **options):
    """One dungeon view with paced search events, then the route and hero walk.

    Use step_seconds=0.7 to hold each search event longer, and explore_frames=120
    to include more recorded snapshots. explore_frames=None shows every stored
    event (use a larger run_search trace_limit for small complete demonstrations).
    show_costs=True labels each entry cost; custom-cost runs do this by default.
    """
    with tempfile.TemporaryDirectory(prefix='catacombs-movie-') as temp:
        path = save_movie(run, Path(temp)/'search.mp4', **options)
        return _video(Path(path).read_bytes(),
                      _movie_caption(run, show_costs=options.get('show_costs')))
