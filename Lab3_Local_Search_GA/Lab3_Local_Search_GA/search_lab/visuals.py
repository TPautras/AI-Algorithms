"""Presentation only: topology checks, charts, and self-contained animations."""
import html
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import IFrame
from .checks import tour_report

COLORS = ['#087f8c', '#d26532', '#7059b8', '#237ab2', '#8c5c3c', '#577c31']
STYLE = {'font.family':'DejaVu Sans', 'font.size':10,
         'axes.spines.top':False, 'axes.spines.right':False,
         'axes.labelcolor':'#43536b', 'text.color':'#182b45',
         'axes.titleweight':'bold', 'figure.facecolor':'white',
         'axes.facecolor':'#f8fafc', 'axes.edgecolor':'#d9e2ec',
         'grid.color':'#e4ebf2', 'grid.alpha':.8}


def show_tour(problem, tour, layout='geometry', focus=None, title='One closed tour'):
    """Fixed irregular geometry by default; focus isolates two true tour edges."""
    tour_report(problem, tour)
    if layout not in ('both', 'geometry', 'circle'):
        raise ValueError('layout must be both, geometry, or circle.')
    if focus is not None and focus not in range(problem.n):
        raise ValueError('focus must be a city ID or None.')
    views = ['geometry', 'circle'] if layout == 'both' else [layout]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(views), figsize=(6.1*len(views), 5.4),
                                 squeeze=False, layout='constrained')
        for ax, view in zip(axes.ravel(), views):
            if view == 'circle':
                theta = np.pi/2 - np.arange(problem.n)*2*np.pi/problem.n
                xy = np.zeros((problem.n, 2))
                xy[np.asarray(tour)] = np.c_[np.cos(theta), np.sin(theta)]
            else:
                xy = problem.xy
            for a, b in problem.edges(tour):
                selected = focus is None or focus in (a,b)
                ax.plot(xy[[a,b],0], xy[[a,b],1], color=COLORS[0] if selected else '#bfcbd6',
                        alpha=1 if selected else .28, lw=2.3 if selected else .9, zorder=2)
            ax.scatter(xy[:,0], xy[:,1], s=64, c='white', edgecolors='white', lw=5, zorder=3)
            ax.scatter(xy[:,0], xy[:,1], s=32, c='white', edgecolors='#182b45', zorder=4)
            for i, (x,y) in enumerate(xy):
                label = str(i) if view == 'circle' else f'{i} {problem.names[i]}'
                ax.annotate(label, (x,y), xytext=(5,6), textcoords='offset points', fontsize=8)
            if focus is not None:
                ax.scatter(*xy[focus], s=110, c='#efb752', edgecolors='#182b45', zorder=5)
            ax.set_title('Schematic coordinates' if view == 'geometry' else 'Circular view · visiting order',
                         loc='left', pad=14)
            ax.set_aspect('equal')
            ax.margins(.22 if view == 'geometry' else .17)
            if view == 'circle':
                ax.axis('off')
                ax.text(0, 0, 'One cycle\nDegree 2 at every city', ha='center', va='center', fontsize=11)
            else:
                ax.grid(True, lw=.6)
                ax.set(xlabel='x · illustrative units', ylabel='y')
        suffix = f' · focus: {problem.names[focus]}' if focus is not None else ''
        fig.suptitle(f'{title} · cost {problem.cost(tour):,.1f}{suffix}', fontsize=14, weight='bold')
        plt.close(fig)
    return fig


def show_network(cities, tour=None):
    """All possible edges, optionally compared with one chosen closed tour."""
    names = tuple(cities)
    xy = np.asarray(list(cities.values()), dtype=float)
    n = len(names)
    if tour is not None:
        a = np.asarray(tour)
        if (a.shape != (n,) or not np.issubdtype(a.dtype, np.integer)
                or not np.array_equal(np.sort(a), np.arange(n))):
            raise ValueError('tour must contain every city ID exactly once.')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 1 if tour is None else 2,
                                 figsize=(11, 7) if tour is None else (14, 6.5),
                                 squeeze=False, layout='constrained')
        for k, ax in enumerate(axes.ravel()):
            for i in range(n):
                for j in range(i+1, n):
                    ax.plot(xy[[i,j],0], xy[[i,j],1], color='#94a3b8',
                            lw=.65, alpha=.27 if k == 0 else .07, zorder=1)
            if k == 1:
                for i, j in zip(tour, np.roll(tour, -1)):
                    ax.plot(xy[[i,j],0], xy[[i,j],1], color=COLORS[0], lw=2, zorder=2)
            ax.scatter(*xy.T, s=78, color='white', edgecolor='#152941', linewidth=1.2, zorder=3)
            ax.scatter(*xy[0], s=130, color='#efb752', edgecolor='#152941', zorder=4)
            for i, (x,y) in enumerate(xy):
                ax.annotate(f'{i} {names[i]}', (x,y), xytext=(6,7), textcoords='offset points',
                            fontsize=8, zorder=5,
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.8))
            ax.set_aspect('equal')
            ax.margins(x=.20, y=.13)
            ax.set(xlabel='x · synthetic units', ylabel='y · synthetic units')
            ax.grid(alpha=.3)
            ax.set_title(f'All choices: {n} cities · {n*(n-1)//2} possible edges' if k == 0
                         else f'One solution: {n} chosen edges · one closed tour', loc='left', pad=16)
        fig.suptitle('A delivery network on an irregular map', fontsize=18, weight='bold')
        plt.close(fig)
    return fig


def show_move(problem, before, after, title='One move changes a complete solution'):
    """Same coordinates in both panels; undirected edge differences are explicit."""
    for tour in (before, after):
        tour_report(problem, tour)
    old = {tuple(sorted(e)) for e in problem.edges(before)}
    new = {tuple(sorted(e)) for e in problem.edges(after)}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12,5.5), layout='constrained')
        for ax, edges, changed, color, label in [
            (axes[0], old, old-new, '#b54735', 'Before · dashed edges are removed'),
            (axes[1], new, new-old, '#087f8c', 'After · bold edges are added'),
        ]:
            for i,j in sorted(edges):
                selected = (i,j) in changed
                ax.plot(problem.xy[[i,j],0], problem.xy[[i,j],1],
                        color=color if selected else '#a8b5c3', lw=3 if selected else 1,
                        linestyle='--' if selected and ax is axes[0] else '-', zorder=2)
            ax.scatter(*problem.xy.T, s=60, color='white', edgecolor='#182b45', zorder=3)
            for i,(x,y) in enumerate(problem.xy):
                ax.annotate(str(i), (x,y), xytext=(6,6), textcoords='offset points', fontsize=9)
            ax.set_aspect('equal'); ax.margins(.17); ax.grid(alpha=.4)
            ax.set_title(label, loc='left', fontsize=11)
            ax.set(xlabel='x · same synthetic coordinates', ylabel='y')
        delta = problem.cost(after)-problem.cost(before)
        fig.suptitle(f'{title}\n{problem.cost(before):.1f} → {problem.cost(after):.1f} · Δ = {delta:+.1f}',
                     fontsize=14, weight='bold')
        plt.close(fig)
    return fig


def plot_trace(run):
    """Static accessible evidence alongside the optional JavaScript animation."""
    if not run.events:
        raise ValueError('This run has no trace: rerun it with record=True.')
    events = run.events
    x = [e['evaluations'] for e in events]
    with plt.rc_context(STYLE):
        fig, (ax,bx) = plt.subplots(1,2,figsize=(12,3.7),layout='constrained')
        ax.plot(x,[e['cost'] for e in events],label='Population best' if run.mode=='ga' else 'Current',color=COLORS[0])
        ax.plot(x,[e['best'] for e in events],label='Best evaluated so far',color=COLORS[2],linestyle='--')
        worse = [e for e in events if e['accepted'] and e['delta'] is not None and e['delta']>0]
        if worse:
            ax.scatter([e['evaluations'] for e in worse],[e['cost'] for e in worse],
                       color=COLORS[1],s=15,label='Accepted worsening move',zorder=4)
        if run.mode == 'sa' and any(e['temperature'] is not None for e in events):
            bx.plot(x,[e['temperature'] for e in events],color=COLORS[1])
            bx.set(ylabel='Temperature',title='Cooling used in this run')
        elif run.mode == 'ga':
            bx.plot(x,[e.get('diversity',np.nan) for e in events],color=COLORS[2])
            bx.set(ylabel='Distinct-cycle fraction',ylim=(0,1.05),title='Exact duplicates in the population')
        else:
            decisions=[e for e in events if e['accepted'] is not None]
            bx.scatter([e['evaluations'] for e in decisions],[e['delta'] for e in decisions],
                       c=[COLORS[0] if e['accepted'] else COLORS[1] for e in decisions],s=9)
            bx.axhline(0,color='#526379',lw=1)
            if run.mode == 'sa':
                bx.set(ylabel='Candidate cost − current cost',title='Δ of each proposal · teal accepted / orange rejected')
            else:
                bx.set(ylabel='Best neighbour cost − current cost',title='Δ of the best neighbour at each step · teal accepted / orange rejected')
        ax.set(ylabel='Tour cost ↓',title=run.name); ax.legend(fontsize=8,frameon=False)
        for target in (ax,bx):
            target.set_xlabel('Objective evaluations'); target.grid(alpha=.4)
        plt.close(fig)
    return fig


def plot_schedules(schedules, steps=1000):
    if steps < 1:
        raise ValueError('steps must be positive.')
    t = np.unique(np.linspace(0, steps-1, min(200,steps)).astype(int))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9,2.8), layout='constrained')
        for i,(label,schedule) in enumerate(schedules.items()):
            values = [schedule(int(k)) for k in t]
            if not np.isfinite(values).all() or min(values)<0:
                raise ValueError('Schedules must return finite, nonnegative temperatures.')
            ax.plot(t, values, label=label, color=COLORS[i%len(COLORS)], lw=2)
        ax.set(xlabel='Proposal index t (starts at 0)', ylabel='Temperature', title='Cooling over max_steps proposals')
        ax.legend(frameon=False)
        ax.grid(True)
        plt.close(fig)
    return fig


def plot_results(results, runs):
    """Cost convergence, final cost and time. IQR is not a confidence interval.

    A run that used fewer evaluations than the longest run keeps its final best value up to the end of the axis.
    This does not claim additional evaluations were performed.
    """
    labels = list(results.method.unique())
    longest = max(len(run.history) for run in runs.values())
    with plt.rc_context(STYLE):
        fig, (ax,bx,tx) = plt.subplots(1,3,figsize=(14.5,4.2), layout='constrained')
        for i,label in enumerate(labels):
            group = results[results.method==label]
            histories = [runs[(label,int(seed))].history for seed in group.seed]
            h = np.asarray([np.pad(a,(0,longest-len(a)),mode='edge') for a in histories])
            x = np.arange(1,longest+1)
            lo,med,hi = np.quantile(h,[.25,.5,.75],axis=0)
            color = COLORS[i%len(COLORS)]
            ax.plot(x,med,label=label,color=color,lw=2)
            ax.fill_between(x,lo,hi,color=color,alpha=.12)
            offsets = np.linspace(-.13,.13,len(group))
            for target,column in [(bx,'best_cost'),(tx,'seconds')]:
                target.scatter(i+offsets,group[column],s=24,color=color,alpha=.75,zorder=3)
                q25,median,q75 = group[column].quantile([.25,.5,.75])
                target.plot([i,i],[q25,q75],color=color,lw=5,alpha=.45)
                target.plot([i-.21,i+.21],[median,median],color='#182b45',lw=2)
        ax.set(xlabel='Objective evaluations',ylabel='Best cost so far ↓',title='Convergence · median + IQR')
        ax.legend(frameon=False,fontsize=8)
        bx.set(ylabel='Final best cost ↓',title='Quality · every run')
        tx.set(ylabel='Elapsed seconds ↓',title='Runtime · snapshots disabled')
        for target in (bx,tx):
            target.set_xticks(range(len(labels)),labels,rotation=22,ha='right')
        for target in (ax,bx,tx):
            target.grid(True,axis='y')
        plt.close(fig)
    return fig


def _player_document(problem, run, max_frames=200, layout='geometry'):
    run.validate()
    if max_frames<2 or layout not in ('geometry','circle'):
        raise ValueError('Use max_frames >= 2 and layout geometry or circle.')
    frames = run.frames
    if not frames:
        raise ValueError('Rerun your algorithm with record=True to create an animation.')
    for frame in frames:
        for field in ('tour','best_tour','candidate'):
            if field in frame:
                tour_report(problem,frame[field])
    if len(frames)>max_frames:
        early = min(25,max_frames//2)
        indices = sorted(set(range(early)) | set(np.linspace(early,len(frames)-1,max_frames-early).astype(int)))
        frames = [frames[i] for i in indices]
    payload = dict(name=run.name,seed=run.seed,metric=getattr(problem,'metric','euclidean'),mode=run.mode,layout=layout,
                   names=list(problem.names),xy=problem.xy.tolist(),frames=frames,
                   events=run.events,total=run.evaluations,steps=run.steps,settings=run.settings)
    assets = Path(__file__).with_name('assets')
    data = json.dumps(payload,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template = (assets/'player.html').read_text(encoding='utf-8')
    return (template.replace('__STYLE__',(assets/'player.css').read_text(encoding='utf-8'))
            .replace('__DATA__',data)
            .replace('__SCRIPT__',(assets/'player.js').read_text(encoding='utf-8')))


def animate_run(problem, run, max_frames=200, layout='geometry'):
    document = _player_document(problem,run,max_frames,layout)
    return IFrame('about:blank',width='100%',height=870,extras=[
        'title="Interactive search trace"','sandbox="allow-scripts"',
        'style="border:0;background:#fff"',
        'srcdoc="'+html.escape(document,quote=True)+'"',
    ])


def save_player(problem, run, filename, layout='geometry'):
    path = Path(filename)
    path.write_text(_player_document(problem,run,layout=layout),encoding='utf-8')
    return str(path.resolve())
