"""Repeated-run and timing support. No algorithm implementation."""
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd


def _random_start(problem, rng):
    """The shared initial state: problem.random_state(rng) when defined (e.g. a Sudoku), else a random tour."""
    return problem.random_state(rng) if hasattr(problem, "random_state") else problem.random_tour(rng)


def benchmark(problem, methods, seeds=range(5), warmup=True):
    """method(problem, seed, initial, record=False) -> finished SearchLog; see as_method(solver, **options).

    Method order is randomized for timing. Methods share a start for each seed.
    Snapshots/animations are disabled. The steps and evaluations actually used are reported.
    """
    rows, runs = [], {}
    order_rng = np.random.default_rng(7310)
    if warmup:
        warm_start = _random_start(problem, np.random.default_rng(8124))
        for method in methods.values():
            method(problem, 8124, warm_start.copy(), record=False).validate()
    for seed in seeds:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2025]))
        initial = _random_start(problem, rng)
        labels = list(methods)
        for index in order_rng.permutation(len(labels)):
            label = labels[int(index)]
            t0 = time.perf_counter()
            run = methods[label](problem, int(seed), initial.copy(), record=False)
            seconds = time.perf_counter() - t0
            run.validate()
            if run.capture:
                raise ValueError(f"{label}: timing runs must be recorded with record=False.")
            runs[(label, int(seed))] = run
            rows.append(dict(method=label, seed=int(seed), best_cost=run.best_cost,
                             steps=run.steps, evaluations=run.evaluations, seconds=seconds,
                             initial_cost=problem.cost(initial),
                             n=getattr(problem, "n", None), metric=getattr(problem, "metric", None),
                             settings=json.dumps(run.settings, sort_keys=True)))
    if not rows:
        raise ValueError("Register at least one implemented method and one seed.")
    results = pd.DataFrame(rows)
    order = {label: i for i, label in enumerate(methods)}
    results["_order"] = results.method.map(order)
    return results.sort_values(["_order", "seed"]).drop(columns="_order").reset_index(drop=True), runs


def summarize(results):
    return results.groupby("method", sort=False).agg(
        runs=("best_cost", "count"), median_cost=("best_cost", "median"),
        q25_cost=("best_cost", lambda x: x.quantile(.25)),
        q75_cost=("best_cost", lambda x: x.quantile(.75)),
        median_seconds=("seconds", "median"),
        q25_seconds=("seconds", lambda x: x.quantile(.25)),
        q75_seconds=("seconds", lambda x: x.quantile(.75)),
        median_steps=("steps", "median"), min_steps=("steps", "min"),
        median_evaluations=("evaluations", "median"),
        min_evaluations=("evaluations", "min"), max_evaluations=("evaluations", "max"),
    ).round(4)


def paired_effect(results, reference, challenger):
    pivot = results.pivot(index="seed", columns="method", values="best_cost")
    pair = pivot[[reference, challenger]].dropna()
    d = pair[challenger] - pair[reference]
    return pd.Series({"paired_runs": len(pair), "median_cost_delta": d.median(),
                      "challenger_wins": int((d < -1e-9).sum()),
                      "ties": int((d.abs() <= 1e-9).sum()),
                      "challenger_losses": int((d > 1e-9).sum())})


def export_results(results, filename="results.csv"):
    path = Path(filename)
    results.to_csv(path, index=False)
    return str(path.resolve())


def environment():
    import matplotlib
    return {"Python": platform.python_version(), "NumPy": np.__version__,
            "pandas": pd.__version__, "Matplotlib": matplotlib.__version__}
