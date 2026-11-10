# The Blind Catacombs · Search lab

Open 01_blind_catacombs.ipynb beside catacombs_lab/, engine/ and maps/. The notebook
is a practical part on search algorithms: a worked example (D1), six implementation
exercises (I1–I6, nine functions to write), written questions (Q1–Q12) and three
experiments (E1–E3). Suggested pacing: two sessions of three hours plus personal work.

## Run

Use Python 3.10+ with JupyterLab or Notebook. From this folder:

```bash
python -m pip install -r requirements-catacombs.txt
python -m jupyterlab
```

Run the first code cell before the imports: it installs missing dependencies into
the Python kernel running the notebook (internet needed the first time). The
introductory clip is a saved movie and plays even before the rendering packages are
installed. Initial Run All shows that clip and runs the supplied DFS on the Gallery;
the cells that call a function you have not written yet raise NotImplementedError
until you complete it. After a lab-file update, rerun the imports cell (it reloads
the helpers) or restart the kernel.

## The model

Problem(initial, goal) provides actions, result, goal_test, action_cost (the cost of
one step) and the cumulative form path_cost(c, state1, action, state2) = c +
action_cost. Node stores state, parent, action, path_cost (g), depth and h; reuse
child_node, expand, path and solution. DungeonPathProblem is the maze domain;
GraphProblem represents small graphs (example_graph() and the reopening fixtures of
Section 7) and records the order of goal tests and expansions, which trace_graph
reports so that an implementation can be checked on a small graph before running on
a maze. The supplied modules contain no search algorithm.

## Launch cells and checks

```python
available_mazes()  # all accepted maze identifiers
run = run_search(maze="the_mire", solver=depth_first_search, name="DFS")
report(run)
display(animate_run(run))
trace_graph(breadth_first_search, example_graph())   # tested/expanded order, plan, cost
```

maze chooses the maze, solver the search function, name the label; for Greedy and A*
add heuristic=manhattan. Each launch cell has one MAZE_NAME setting (E1 a MAZE_NAMES
list); every baseline in the cell reruns on that maze. check_run validates a returned
route and its measurements or reports the failure/cutoff/budget outcome;
check_comparison verifies that runs describe the same problem and returns True when
all are solved; check_heuristic lists consistency violations. A work or depth limit
is reported as such, never as a proof that no route exists.

Search functions return a plan list, including [] when the start is the goal, or
None after exhaustive failure; depth-limited search adds CUTOFF. The printed report
gives moves, cost, expansions (calls to actions) and the peak number of entries held
in the frontier; reached sets and tables are not counted.

## Costs and animations

Without a custom function, terrain costs are 1 (stone), 2 (gravel) or 5 (mire), paid
on entering a cell. unit_cost, random_costs(maze, seed=7, low=1, high=9) and
with_costs(maze, cost_function) build alternative cost models; give every approach
the same costed maze when comparing. Corner badges such as c7 give an entry cost;
show_costs=True displays them.

Animations use the dungeon renderer: violet exploration, cyan ranks on waiting
cells, a thin white line for the path to the current node, the gold route, then
the hero's walk. Depth-limited runs show depth and limit, an orange outline at a
cutoff, BEGIN PASS and PASS ENDED between passes. Use animate_run(run,
step_seconds=0.7, explore_frames=120) for a slower movie with more snapshots, and
save_movie(run, "my_search.mp4") to export. Long traces are sampled and gaps are
marked. Open Catacombs_Animation_Demo.html to view the introduction by itself.

## Submission

Submit the completed notebook with outputs and written answers, your names, and the
required files, through the channel specified for the course. Restart the kernel and
Run All before submitting; keep the folder structure in your ZIP.
With every animation rendered, the completed notebook weighs about 20 to 25 MB (each
animation embeds a short video); check that the channel accepts this size.
