"""Problem models, passive search accounting and reproducible experiment support.

No DFS/BFS/UCS/Greedy/A* implementation is supplied in this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from itertools import count
import heapq
import math
import time
import platform
import numpy as np
import pandas as pd
from engine.core.grid import DungeonMap
from engine.core.problem import CUTOFF, SearchBudgetExceeded, replay
from .models import Problem, Node, DungeonPathProblem, GraphProblem

MAPS = {name: Path(__file__).resolve().parents[1]/'maps'/floor/(name+'.txt')
        for floor,names in [('floor1',('first_dark','flooded_gallery','the_mire',
                                      'endless_stair','sealed_cell','twin_doors')),
                            ('floor2',('lodestone','star_road','workshop','oracle_vault')),
                            ('lab',('four_chambers','canal_bridges','switchback_courts','recursion_room'))]
        for name in names}


def available_mazes():
    """Names accepted by load_map(name) and run_search(maze=name, solver=...)."""
    return tuple(MAPS)


def load_map(name):
    """Resolve a registered name, or retain an existing custom DungeonMap."""
    if isinstance(name,DungeonMap):
        return name
    if not isinstance(name,str):
        raise TypeError('Choose a registered maze name or a DungeonMap object.')
    if name not in MAPS:
        raise ValueError(f'Unknown maze {name!r}; choose {available_mazes()}')
    return DungeonMap.load(MAPS[name])


def reconstruct(node):
    """Compatibility helper; the notebook uses node.solution()."""
    return node.solution()


def node_path(node):
    return [] if node is None else [entry.state for entry in node.path()]


class Stack:
    """LIFO, with snapshot() in the actual next-pop order."""
    def __init__(self): self._items=[]
    def push(self,item): self._items.append(item)
    def pop(self): return self._items.pop()
    def __len__(self): return len(self._items)
    def is_empty(self): return not self._items
    def snapshot(self): return [(node,None) for node in reversed(self._items)]


class Queue:
    """FIFO. Queue operations themselves are O(1)."""
    def __init__(self): self._items=deque()
    def push(self,item): self._items.append(item)
    def pop(self): return self._items.popleft()
    def __len__(self): return len(self._items)
    def is_empty(self): return not self._items
    def snapshot(self): return [(node,None) for node in self._items]


class PriorityQueue:
    """Minimum priority, FIFO on ties, lazy duplicate entries allowed."""
    def __init__(self): self._heap=[]; self._sequence=count()
    def push(self,item,priority):
        if not math.isfinite(priority): raise ValueError('Priority must be finite.')
        heapq.heappush(self._heap,(float(priority),next(self._sequence),item))
    def pop(self): return heapq.heappop(self._heap)[2]
    def pop_with_priority(self):
        priority,_,item=heapq.heappop(self._heap)
        return item,priority
    def __len__(self): return len(self._heap)
    def is_empty(self): return not self._heap
    def snapshot(self): return [(node,priority) for priority,_,node in sorted(self._heap)]


class MeasuredProblem(Problem):
    """Count real successor-list requests and generated successor transitions."""
    def __init__(self,inner,budget):
        super().__init__(inner.initial, inner.goal)
        self.inner=inner
        self.budget=budget
        self.expansions=0
        self.generated=0
        self.expanded=set()
        self.expanded_in_pass=set()  # Passive display state; IDS work stays cumulative.
        self.generated_states={inner.initial}
    def goal_test(self,state): return self.inner.goal_test(state)
    def action_cost(self,state,action,next_state): return self.inner.action_cost(state,action,next_state)
    def path_cost(self,c,state1,action,state2): return self.inner.path_cost(c,state1,action,state2)
    def value(self,state): return self.inner.value(state)
    def actions(self,state):
        if self.expansions>=self.budget:
            raise SearchBudgetExceeded(f'Expansion ceiling {self.budget} reached.')
        self.expansions+=1
        self.expanded.add(state)
        self.expanded_in_pass.add(state)
        return self.inner.actions(state)
    def result(self,state,action):
        child=self.inner.result(state,action)
        self.generated+=1
        self.generated_states.add(child)
        return child


def _sample_frames(frames, limit):
    """Keep the opening, pass restarts and end when thinning recorded events."""
    if len(frames) <= limit:
        return list(frames)
    head = set(range(min(12, limit // 3)))
    last_in_pass = {}
    for i, frame in enumerate(frames):
        if frame.get('depth_limit') is not None and frame['event'] != 'limit':
            last_in_pass[frame['depth_limit']] = i
    anchors = sorted(({i for i, frame in enumerate(frames)
                       if frame['event'] in ('limit', 'pass_end', 'goal', 'failure', 'budget')}
                      | set(last_in_pass.values())) - head)
    tail = {len(frames)-1}
    room = limit-len(head | tail)
    if len(anchors) > room:
        anchors = [anchors[i] for i in np.linspace(0, len(anchors)-1, room).astype(int)]
    selected = head | set(anchors) | tail
    rest = [i for i in range(len(frames)) if i not in selected]
    room = limit-len(selected)
    if room and rest:
        selected.update(rest[i] for i in np.linspace(0, len(rest)-1, room).astype(int))
    return [frames[i] for i in sorted(selected)]


def _best_cost(table,node):
    """Best known g for node.state in a reached table of Nodes or of numbers."""
    value=table.get(node.state,node.path_cost)
    return getattr(value,'path_cost',value)


class Trace:
    """Observe frontier/path snapshots; never choose, reject or update a node."""
    def __init__(self,problem,record=True,memory_label='Frontier entries',trace_limit=260):
        self.problem=problem
        self.enabled=record
        self.memory_label=memory_label
        self.peak_frontier=0
        self.frames=[]
        self.step=0
        self._last=None
        self._pending=None
        self._stride=1
        self._pass_start=0
        self.trace_limit=trace_limit
    def capture(self,node,frontier=None,*,event='select',best_g=None,depth_limit=None,note='',
                return_from=None,return_status=None,pending_calls=None):
        self.step+=1
        size=0 if frontier is None else len(frontier)
        self.peak_frontier=max(self.peak_frontier,size)
        if event == 'limit':
            self.problem.expanded_in_pass.clear()
            self._pass_start=self.problem.expansions
        if not self.enabled:return
        self._pending=(node,frontier,event,best_g,depth_limit,note,return_from,return_status,pending_calls)
        # Keep early decisions, limit changes and the latest observation; thin long traces.
        important=event in ('goal','failure','limit','pass_end','budget')
        if (self.problem.expansions<20 or self.problem.expansions%self._stride==0 or important):
            frame=self._make_frame(*self._pending)
            self.frames.append(frame)
            self._last=frame
            # Limit snapshots (not the exact scalar accounting) on long IDS traces.
            if len(self.frames)>max(600, 2*self.trace_limit):
                self.frames=_sample_frames(self.frames, max(320, self.trace_limit))
                self._stride*=2
    def _make_frame(self,node,frontier,event,best_g,depth_limit,note,return_from=None,return_status=None,
                    pending_calls=None):
        if frontier is None: entries=[]
        elif hasattr(frontier,'snapshot'):entries=frontier.snapshot()
        else:entries=[(n,None) for n in reversed(frontier)]
        records=[dict(state=n.state,g=float(n.path_cost),h=float(n.h),f=float(n.f),
                      priority=priority,depth=n.depth,
                      stale=bool(best_g is not None and n.path_cost!=_best_cost(best_g,n)))
                 for n,priority in entries]
        kind=('stack' if isinstance(frontier, Stack) else 'queue' if isinstance(frontier, Queue)
              else 'priority' if isinstance(frontier, PriorityQueue) else 'path')
        # Copy suspended successor loops, deepest call first. Do not generate
        # children, call the problem or change the solver's measured work.
        deferred = None if pending_calls is None else [
            dict(parent=parent.state, path=node_path(parent), actions=list(actions[index:]))
            for parent, actions, index in reversed(pending_calls)]
        return dict(step=self.step,expansions=self.problem.expansions,
                    generated=self.problem.generated,unique_expanded=len(self.problem.expanded),
                    current=None if node is None else node.state,frontier=records,
                    frontier_size=len(entries),explored=sorted(self.problem.expanded),
                    frontier_kind=kind,pass_explored=sorted(self.problem.expanded_in_pass),
                    pass_expansions=self.problem.expansions-self._pass_start,
                    stale_current=bool(node is not None and best_g is not None
                                       and node.path_cost != _best_cost(best_g,node)),
                    path=node_path(node),g=0.0 if node is None else float(node.path_cost),
                    h=0.0 if node is None else float(node.h),f=0.0 if node is None else float(node.f),
                    depth=0 if node is None else node.depth,note=note,depth_limit=depth_limit,event=event,
                    return_from=return_from,return_status=return_status,deferred_actions=deferred)
    def finish(self,status,path):
        if not self.enabled:return
        if self._pending is not None:
            frame=self._make_frame(*self._pending)
            frame['event']='goal' if status=='solved' else status
            frame['note']=f'Search finished: {status}. '+frame['note']
            if status=='solved':frame['path']=path
        else:
            frame=self._make_frame(None,None,status,None,None,f'Search finished: {status}.')
            frame['path']=path
        self.frames.append(frame)
        if len(self.frames)>self.trace_limit:
            self.frames=_sample_frames(self.frames, self.trace_limit)


def record_step(log,node,frontier=None,*,event='select',best_g=None,reached=None,depth_limit=None,note='',
                return_from=None,return_status=None,pending_calls=None):
    """Record a search observation. With log=None, this is a no-op.

    node is the current Node. frontier is a supplied container, or the list
    of active recursive nodes in root-to-current order. Its length is used
    for the peak-storage counter. The recorder does not choose or update nodes.

    event names describe the observation: 'select' (current node), 'expand'
    (updated alternatives), 'goal', 'cutoff', or 'backtrack'. For a recorded
    return, node is the parent, return_from is the child's state, and
    return_status is 'cutoff' or 'failure'. depth_limit is the active bound.

    'limit' marks the beginning of a bounded pass, with its root, an empty
    frontier and depth_limit. 'pass_end' marks an unsuccessful pass ending,
    with node=None, an empty frontier and return_status. These events delimit
    pass colours and counters; each belongs to the bounded traversal itself.

    pending_calls describes suspended successor loops, ordered root to deepest
    call. Each entry is (parent_node, actions, next_index), where next_index
    identifies the first untried action. The recorder copies this information
    to display deferred alternatives; it does not generate successors.

    reached (alias best_g) is an optional mapping from a state to the best
    known node or cost for that state. It is used
    only to mark stale queue entries. note is an optional caption; a note
    beginning with 'Initialize' identifies the initial frontier snapshot.
    """
    if reached is not None:
        best_g=reached
    if log is not None:
        log.capture(node,frontier,event=event,best_g=best_g,depth_limit=depth_limit,note=note,
                    return_from=return_from,return_status=return_status,pending_calls=pending_calls)


@dataclass
class SearchRun:
    name: str
    dungeon: DungeonMap
    status: str
    plan: object
    cost: object
    length: object
    expansions: int
    generated: int
    unique_expanded: int
    peak_frontier: int
    memory_label: str
    seconds: float
    frames: list
    path: list
    settings: dict

    def summary(self):
        return dict(map=self.dungeon.name,method=self.name,status=self.status,cost=self.cost,
                    length=self.length,expansions=self.expansions,generated=self.generated,
                    unique_expanded=self.unique_expanded,peak_frontier=self.peak_frontier,
                    memory_label=self.memory_label,seconds=self.seconds)


def run_search(dungeon=None,solver=None,*,maze=None,heuristic=None,budget=50000,record=True,name=None,
               memory_label=None,cost_function=None,trace_limit=260,**solver_options):
    """Run a solver on a named maze or an existing DungeonMap.

    Example: run_search(maze="the_mire", solver=depth_first_search, name="DFS").
    The original run_search(dungeon, solver) call also accepts a maze-name string.
    name labels the approach; maze selects the map. Loading precedes timing.
    cost_function((x,y)) sets fixed positive destination costs before timing.
    Use with_costs first when several approaches must share the same cost table.
    trace_limit bounds retained observations; increase it for detailed small demos.
    """
    if maze is not None:
        if dungeon is not None:
            raise ValueError('Choose one maze: use either dungeon or maze, not both.')
        dungeon=maze
    if isinstance(dungeon,str):
        dungeon=load_map(dungeon)
    if not isinstance(dungeon,DungeonMap):
        raise TypeError('Provide a maze name or DungeonMap; use available_mazes() to list names.')
    if not callable(solver):
        raise TypeError('Provide a search function, for example solver=depth_first_search.')
    if not isinstance(budget,int) or isinstance(budget,bool) or budget<1:
        raise ValueError('budget must be a positive integer')
    if not isinstance(trace_limit,int) or isinstance(trace_limit,bool) or trace_limit<16:
        raise ValueError('trace_limit must be an integer >= 16.')
    if cost_function is not None:
        from .costs import with_costs
        dungeon=with_costs(dungeon,cost_function)
    p=MeasuredProblem(DungeonPathProblem(dungeon),budget)
    label=name or solver.__name__
    if memory_label is None:
        memory_label='Active path nodes' if 'iterative_deepening_a_star' in solver.__name__ else 'Frontier entries'
    trace=Trace(p,record,memory_label,trace_limit)
    start=time.perf_counter()
    try:
        plan=solver(p,log=trace,**solver_options) if heuristic is None else solver(p,heuristic,log=trace,**solver_options)
        status='cutoff' if plan==CUTOFF else 'failure' if plan is None else 'solved'
    except SearchBudgetExceeded:
        plan=None
        status='budget'
    seconds=time.perf_counter()-start
    path=[];cost=None;length=None
    if status=='solved':
        report=replay(dungeon,plan)
        if not report['ok']:raise AssertionError(f'{label}: invalid returned plan: {report["reason"]}')
        path=report['states'];cost=report['cost'];length=report['length']
    trace.finish(status,path)
    return SearchRun(label,dungeon,status,plan,cost,length,p.expansions,p.generated,len(p.expanded),
                     trace.peak_frontier,memory_label,seconds,trace.frames,path,
                     dict(budget=budget,trace_limit=trace_limit,
                          cost_model=getattr(dungeon,'cost_label','terrain'),
                          heuristic=getattr(heuristic,'__name__','none'),
                          solver=getattr(solver,'__name__','custom'),**solver_options))


def check_solution(run,*,expected_cost=None,expected_length=None):
    """Require a legal solution. Optional expected values are explicit fixture checks."""
    label=f'{run.dungeon.name} | {run.name}'
    assert run.status=='solved',f'{label}: {run.status}, no validated solution; use check_run for any outcome.'
    report=replay(run.dungeon,run.plan)
    assert report['ok'],f'{label}: {report["reason"]}'
    assert run.length==report['length'] and np.isclose(run.cost,report['cost']),f'{label}: recorded cost/length disagrees with the route.'
    assert run.path==report['states'],f'{label}: recorded path disagrees with the route.'
    if expected_cost is not None:
        assert np.isclose(run.cost,expected_cost),(
            f'{label}: got cost {run.cost}, expected {expected_cost}. '
            'An explicit expected value must belong to this maze.')
    if expected_length is not None:
        assert run.length==expected_length,(
            f'{label}: got {run.length} moves, expected {expected_length}. '
            'An explicit expected value must belong to this maze.')
    return f'PASS · {label}: {run.length} moves, cost {run.cost}, {run.expansions} expansions'


def check_run(run):
    """Validate a returned route or the structure of a reported non-solution.

    No search is performed here. A solver's failure claim is not independently
    proved, and cutoff/budget outcomes are never mistaken for solutions.
    """
    if run.status=='solved':
        return check_solution(run)
    label=f'{run.dungeon.name} | {run.name}'
    assert run.status in ('failure','cutoff','budget'),f'{label}: unknown status {run.status!r}'
    expected_plan=CUTOFF if run.status=='cutoff' else None
    assert run.plan==expected_plan,f'{label}: plan contradicts status {run.status}'
    assert run.cost is None and run.length is None and not run.path,f'{label}: no solution should have no route, cost or length.'
    meanings={
        'failure':'solver reports exhaustive failure; reachability is not independently checked',
        'cutoff':'depth limit reached; no conclusion about reachability',
        'budget':'expansion budget reached; no conclusion about reachability',
    }
    return f'STATUS · {label}: {meanings[run.status]}'


def check_comparison(*runs):
    """Validate same-problem runs and return True only when all are solved.

    Compare actual geometry, start, goal and entry costs, not map display names.
    Contradictory solved/failure claims raise; cutoff and budget are inconclusive.
    This is a cross-check of supplied runs, not an optimal-path oracle.
    """
    if not runs:
        raise ValueError('Provide at least one run to compare.')
    def problem_key(dungeon):
        return (tuple(dungeon.rows),dungeon.start,dungeon.goal,
                tuple(dungeon.terrain_cost(*s) for s in dungeon.floor_positions()))
    expected=problem_key(runs[0].dungeon)
    for run in runs:
        assert problem_key(run.dungeon)==expected,(
            'Comparison inputs differ: rerun every approach with the same maze, '
            'start, goal and terrain costs.')
        check_run(run)
    statuses={run.status for run in runs}
    assert not {'solved','failure'}<=statuses,(
        'Searches disagree on reachability: one returns a solution and another '
        'reports exhaustive failure. Check their implementations.')
    return statuses=={'solved'}


def check_heuristic(dungeon,h):
    """Check finite/nonnegative/goal-zero values and every directed consistency inequality.

    dungeon is a DungeonMap, or a Problem exposing states() such as GraphProblem.
    Conditions: h(n) >= 0, h(goal) = 0 and h(n) <= c(n, n') + h(n') on every edge.
    Consistency implies admissibility.
    """
    if isinstance(dungeon,DungeonMap):
        p=DungeonPathProblem(dungeon);states=dungeon.floor_positions()
    else:
        p=dungeon;states=list(p.states())
    goal=p.goal
    values={s:float(h(s,goal)) for s in states}
    violations=[]
    for s,value in values.items():
        if not np.isfinite(value) or value<0:
            violations.append(dict(state=s,successor=None,reason='nonfinite or negative h'))
        for action in p.actions(s):
            child=p.result(s,action)
            if value>p.path_cost(0,s,action,child)+values[child]+1e-9:
                violations.append(dict(state=s,successor=child,reason='inconsistent edge'))
    if goal in values and not np.isclose(values[goal],0):violations.append(dict(state=goal,successor=None,reason='h(goal) != 0'))
    return pd.DataFrame(violations,columns=['state','successor','reason'])


def benchmark(dungeons,methods,*,repeats=5,budget=50000):
    """Deterministic methods: repetitions measure timing noise, not search robustness.

    methods maps names to (solver, heuristic or None, options dict). Snapshots off.
    Build maps/heuristics before timing. Warm up each pair and shuffle run order.
    """
    if not isinstance(repeats,int) or repeats<1:raise ValueError('repeats must be positive')
    dungeons=list(dungeons)
    rows=[];runs={};rng=np.random.default_rng(3101)
    for dungeon in dungeons:
        names=list(methods)
        for name,(solver,h,options) in methods.items():
            run_search(dungeon,solver,heuristic=h,budget=budget,record=False,name=name,**options)
        for repeat in range(repeats):
            for idx in rng.permutation(len(names)):
                name=names[int(idx)];solver,h,options=methods[name]
                run=run_search(dungeon,solver,heuristic=h,budget=budget,record=False,name=name,**options)
                row=run.summary();row['repeat']=repeat;rows.append(row)
                runs[(dungeon.name,name)]=run
    if not rows:raise ValueError('At least one map and method is required')
    results=pd.DataFrame(rows)
    # Randomise execution order for timing; retain the declared comparison order in figures.
    map_order={d.name:i for i,d in enumerate(dungeons)}
    method_order={name:i for i,name in enumerate(methods)}
    results['_map_order']=results['map'].map(map_order)
    results['_method_order']=results['method'].map(method_order)
    results=results.sort_values(['_map_order','_method_order','repeat']).drop(
        columns=['_map_order','_method_order']).reset_index(drop=True)
    return results,runs


def summarize(results):
    return results.groupby(['map','method'],sort=False).agg(
        status=('status','first'),cost=('cost','first'),moves=('length','first'),
        expansions=('expansions','first'),generated=('generated','first'),
        distinct=('unique_expanded','first'),peak=('peak_frontier','first'),
        memory_label=('memory_label','first'),median_seconds=('seconds','median'),
        q25_seconds=('seconds',lambda x:x.quantile(.25)),
        q75_seconds=('seconds',lambda x:x.quantile(.75))).round(5)


def environment():
    import matplotlib
    return {'Python':platform.python_version(),'NumPy':np.__version__,
            'pandas':pd.__version__,'Matplotlib':matplotlib.__version__}
