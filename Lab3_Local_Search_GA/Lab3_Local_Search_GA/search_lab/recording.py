"""Passive evaluation accounting, archives, and visualization records.

Two layers:

* ``SearchLog`` is the run object: it counts objective calls and steps, archives the best state, and stores the
  events and frames that the figures and the player read. It makes no search decision.
* ``run_search`` runs a solver written like pseudo-code. The solver receives a *recorded problem*: the same
  problem, except that ``cost`` counts one evaluation and archives the best state, that ``neighbors`` and
  ``random_neighbor`` tell the recorder which state is current, and that ``snapshot`` records a population.
  From these calls the recorder infers the current state, the candidate, Δ and the decision of every step,
  so the algorithm itself contains no logging code. The solver stops by itself (``max_steps`` or its own rule)
  and returns nothing.
"""
import inspect
import math
import time
import numpy as np
from .checks import cycle_diversity

_FRAME_CAP = 600          # a long run keeps at most about this many animation frames (the first 60 are always kept)


def _diversity(problem, population):
    """problem.diversity(population) when defined; the distinct-cycle fraction for tours (permutations);
    otherwise the fraction of distinct states (exact duplicates removed)."""
    measure = getattr(problem, "diversity", None)
    if callable(measure):
        return float(measure(population))
    first = np.asarray(population[0])
    if (first.ndim == 1 and np.issubdtype(first.dtype, np.integer)
            and np.array_equal(np.sort(first), np.arange(first.size))):
        return cycle_diversity(population)
    return len({np.asarray(state).tobytes() for state in population}) / len(population)


class SearchLog:
    """evaluate counts objective calls and archives the best tour; record stores observations; no search decisions.

    ``evaluations`` counts the objective calls, ``steps`` the steps of the solver (one call of ``neighbors`` or
    ``random_neighbor``, or one generation after the initial population). record=False disables the snapshots
    for repeated timing experiments.
    """
    def __init__(self, problem, seed=0, label="Search", record=True, settings=None, mode="local"):
        if mode not in ("local", "sa", "ga", "demo"):
            raise ValueError("mode must be local, sa, ga, or demo.")
        self.problem = problem
        self.seed, self.name, self.mode = int(seed), str(label), mode
        self.capture = bool(record)
        self.settings = dict(settings or {})
        self.evaluations, self.steps = 0, 0
        self.best_cost, self._best_tour = float("inf"), None
        self._history, self.events, self.frames = [], [], []
        self._stride = 1
        self._last_frame = None
        self.finished = False
        self.started = time.perf_counter()

    @property
    def best_tour(self):
        return None if self._best_tour is None else self._best_tour.copy()

    @property
    def history(self):
        return np.asarray(self._history, dtype=float)

    def evaluate(self, tour):
        if not self.problem.is_valid(tour):
            raise ValueError("Cannot evaluate an invalid tour.")
        cost = float(self.problem.cost(tour))
        if not np.isfinite(cost):
            raise ValueError("The objective must be finite.")
        self.evaluations += 1
        if cost < self.best_cost:
            self.best_cost, self._best_tour = cost, np.array(tour, copy=True)
        self._history.append(self.best_cost)
        return cost

    def record(self, tour, cost, *, candidate=None, candidate_cost=None,
               accepted=None, delta=None, temperature=None, probability=None,
               population=None, costs=None, generation=None, note=""):
        if not self.capture:
            return
        if not self.evaluations:
            raise ValueError("Evaluate a tour before recording a state.")
        if not self.problem.is_valid(tour):
            raise ValueError("The recorded current tour is invalid.")
        if not np.isfinite(cost):
            raise ValueError("Recorded cost must be finite.")
        if candidate is not None:
            if not self.problem.is_valid(candidate) or candidate_cost is None:
                raise ValueError("A candidate needs a valid tour and its evaluated cost.")
            if not np.isfinite(candidate_cost):
                raise ValueError("Candidate cost must be finite.")
        if temperature is not None and (not np.isfinite(temperature) or temperature < 0):
            raise ValueError("Record a finite, nonnegative temperature.")
        if probability is not None and not 0 <= probability <= 1:
            raise ValueError("Acceptance probability must lie in [0, 1].")
        event = dict(evaluations=self.evaluations, cost=float(cost), best=self.best_cost,
                     accepted=None if accepted is None else bool(accepted),
                     delta=None if delta is None else float(delta),
                     temperature=None if temperature is None else float(temperature),
                     probability=None if probability is None else float(probability),
                     generation=None if generation is None else int(generation), note=str(note))
        if population is not None:
            if costs is None or len(population) != len(costs) or not len(population):
                raise ValueError("Provide a nonempty population and matching cached costs.")
            if any(not self.problem.is_valid(t) for t in population):
                raise ValueError("The recorded population contains an invalid tour.")
            event["diversity"] = _diversity(self.problem, population)
        self.events.append(event)
        frame = dict(event, tour=np.asarray(tour, dtype=int).tolist(),
                     best_tour=self.best_tour.tolist())
        if candidate is not None:
            frame.update(candidate=np.asarray(candidate, dtype=int).tolist(),
                         candidate_cost=float(candidate_cost))
        if population is not None:
            frame["population_costs"] = [float(value) for value in costs]
        self._last_frame = frame
        if len(self.events) <= 60 or population is not None or len(self.events) % self._stride == 0:
            self.frames.append(frame)
            if len(self.frames) > _FRAME_CAP:          # a long run: keep one frame in two after the first 60
                self.frames = self.frames[:60] + self.frames[60::2]
                self._stride *= 2

    def finish(self):
        if not self.evaluations:
            raise ValueError("No tours were evaluated.")
        if self.capture:
            if self._last_frame is None or self._last_frame["evaluations"] != self.evaluations:
                raise ValueError("Record the final state before finish().")
            if not self.frames or self.frames[-1] is not self._last_frame:
                self.frames.append(self._last_frame)
        self.finished = True
        self.seconds = time.perf_counter() - self.started
        return self.validate()

    def validate(self):
        if not self.finished:
            raise ValueError("The run is not finished: run the solver with run_search.")
        assert self.evaluations >= 1 and self.steps >= 0
        assert len(self.history) == self.evaluations
        assert np.all(np.diff(self.history) <= 1e-9), "Best-so-far must not increase."
        assert self.problem.is_valid(self.best_tour)
        assert np.isclose(self.problem.cost(self.best_tour), self.best_cost)
        if self.capture:
            for frame in self.frames:
                assert np.isclose(self.problem.cost(np.asarray(frame["tour"])), frame["cost"]), "State/cost mismatch."
                if "candidate" in frame:
                    assert np.isclose(self.problem.cost(np.asarray(frame["candidate"])),
                                      frame["candidate_cost"]), "Candidate/cost mismatch."
            assert self.frames[-1]["evaluations"] == self.evaluations
        return self


# ------------------------------------------------------------------ the recorded problem and the runner
class RecordedProblem:
    """The problem handed to a solver by ``run_search``: the same object, with the accounting built in.

    * ``cost(state)`` counts one objective evaluation and archives the best state;
    * ``neighbors(state, ...)`` and ``random_neighbor(state, rng, ...)`` run the problem's own methods, count one
      step and tell the recorder that ``state`` is the current one: the states evaluated until the next such call
      are its candidates, the best of them is the recorded candidate, and the search *accepted* it when the next
      current state differs;
    * ``random_tour(rng)`` / ``random_state(rng)`` drawn after the search started mark a random restart;
    * ``snapshot(population)`` records a population (its best member, its costs, its diversity): the one call a
      population-based algorithm has to make, after the initial population and after every generation (each one
      after the first counts one step). ``snapshot(state)`` marks a single current state for algorithms that use
      neither ``neighbors`` nor ``random_neighbor``.

    Every other attribute (``n``, ``names``, ``xy``, ``distances``, ``move_pairs``, ``apply_move``, ``is_valid``,
    ``edges`` ...) is the problem's own.
    """

    def __init__(self, problem, log):
        self._problem, self._log = problem, log
        self._current, self._current_cost = None, None
        self._candidate, self._candidate_cost = None, math.inf       # best state evaluated since the last marker
        self._scanned, self._expected = 0, None                      # evaluations since the marker, scan size
        self._pending_start = None                                   # random state drawn mid-run: a restart
        self._markers, self._generations = set(), 0                  # population snapshots taken so far

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def __repr__(self):
        return f"RecordedProblem({self._problem!r}, {self._log.evaluations} evaluations, {self._log.steps} steps)"

    @property
    def evaluations(self):
        return self._log.evaluations

    # ---- the four calls the recorder listens to
    def cost(self, state):
        """Cost of a state: counts one objective evaluation and archives the best state."""
        value = self._log.evaluate(state)
        if not self._log.capture:
            return value
        if self._current is None:
            self._set_current(state, value, "Initial state")
        elif self._pending_start is not None and np.array_equal(state, self._pending_start):
            self._pending_start = None
            if self._scanned:
                self._close_step(None, self._rejection_note())
            self._set_current(state, value, "Random restart")
        elif not np.array_equal(state, self._current):
            self._scanned += 1
            if value < self._candidate_cost:
                self._candidate, self._candidate_cost = np.array(state, copy=True), value
        return value

    def neighbors(self, state, *args, **kwargs):
        """The problem's complete neighbourhood of ``state``, which becomes the current state (one step)."""
        found = list(self._problem.neighbors(state, *args, **kwargs))
        self._count_step()
        self._mark("neighbors", state, expected=len(found))
        return found

    def random_neighbor(self, state, rng, *args, **kwargs):
        """One random neighbour of ``state``, which becomes the current state (one step)."""
        self._count_step()
        self._mark("random_neighbor", state, expected=1)
        return self._problem.random_neighbor(state, rng, *args, **kwargs)

    def random_tour(self, rng):
        return self._fresh(self._problem.random_tour(rng))

    def random_state(self, rng):
        return self._fresh(self._problem.random_state(rng))

    def snapshot(self, states, note=""):
        """Tell the recorder where the search stands: a population (list of states) after the initial population
        and after each generation, or one state (the current one) for algorithms that use neither neighbors nor
        random_neighbor."""
        if self._problem.is_valid(states):
            self._count_step()
            if self._log.capture:
                self._mark("snapshot", states, expected=None)
            return
        if self._generations:                           # every population after the initial one: one generation
            self._log.steps += 1
        self._generations += 1
        if not self._log.capture:
            return
        population = [np.asarray(state) for state in states]
        if not population:
            raise ValueError("snapshot needs one state or a non-empty population.")
        costs = [float(self._problem.cost(state)) for state in population]       # known costs: not evaluations
        champion = int(np.argmin(costs))
        self._log.record(population[champion], costs[champion], population=population, costs=costs,
                         generation=self._generations - 1, note=note)
        self._markers.add("snapshot")
        self._current, self._current_cost = np.array(population[champion], copy=True), costs[champion]
        self._reset_scan()

    # ---- inference
    @property
    def _stepping(self):
        """True while the trace is read from neighbors / random_neighbor markers (not a population search)."""
        return bool(self._markers & {"neighbors", "random_neighbor", "snapshot"}) and not self._generations

    @property
    def _sampling(self):
        return "random_neighbor" in self._markers and "neighbors" not in self._markers

    def _count_step(self):
        if not self._generations:                       # inside a population search, a generation is the step
            self._log.steps += 1

    def _fresh(self, state):
        if self._log.capture and self._stepping:        # drawn after the search started: a restart once evaluated
            self._pending_start = np.array(state, copy=True)
        return state

    def _mark(self, kind, state, expected):
        self._markers.add(kind)
        if not self._log.capture or self._generations:  # a population search: markers do not define steps
            return
        if self._current is None:
            raise ValueError("Evaluate your starting state with problem.cost(...) before exploring its neighbours.")
        if self._scanned:
            self._close_step(state)
        elif not np.array_equal(state, self._current):   # moved to a state whose cost was already known
            self._set_current(state, float(self._problem.cost(state)), "")
        self._expected = expected

    def _complete(self):
        return self._expected is None or self._scanned >= self._expected

    def _rejection_note(self):
        if self._expected is None or self._sampling or not self._complete():
            return ""
        return "No improving neighbour"

    def _set_current(self, state, cost, note):
        self._current, self._current_cost = np.array(state, copy=True), float(cost)
        self._log.record(self._current, self._current_cost, note=note)
        self._reset_scan()

    def _reset_scan(self):
        self._candidate, self._candidate_cost, self._scanned = None, math.inf, 0

    def _close_step(self, next_state, note=""):
        """Record the step that ends now: the best candidate of the scan, Δ, and whether the search moved."""
        candidate, candidate_cost = self._candidate, self._candidate_cost
        if next_state is None:                                     # restart or end of the run: not accepted
            accepted, state, cost = False, self._current, self._current_cost
        else:
            accepted = not np.array_equal(next_state, self._current)
            if not accepted:
                state, cost = self._current, self._current_cost
            elif np.array_equal(next_state, candidate):
                state, cost = candidate, candidate_cost
            else:                                                  # moved to another scanned state (tie, first improvement)
                state = candidate = np.array(next_state, copy=True)
                cost = candidate_cost = float(self._problem.cost(next_state))
        self._log.record(state, cost, candidate=candidate, candidate_cost=candidate_cost,
                         delta=candidate_cost - self._current_cost, accepted=accepted, note=note)
        self._current, self._current_cost = np.array(state, copy=True), float(cost)
        self._reset_scan()

    def _finish(self):
        log = self._log
        if log.capture:
            if self._scanned:
                # The decision that followed the last evaluation was not observed. A sampling search accepts every
                # improvement or tie; a greedy climber moves after an improving scan.
                delta = self._candidate_cost - self._current_cost
                accepted = delta <= 0 if self._sampling else delta < 0
                if accepted:
                    self._close_step(self._candidate)
                else:
                    self._close_step(None, self._rejection_note())
            elif log._last_frame is None or log._last_frame["evaluations"] != log.evaluations:
                if self._stepping:
                    log.record(self._current, self._current_cost, note="Final state")
                else:
                    log.record(log.best_tour, log.best_cost, note="Final state (best archived)")
        log.mode = "ga" if self._generations else ("sa" if self._sampling else "local")
        return log.finish()


def _plain(value):
    """A JSON-serialisable copy of one setting."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _settings(solver, options):
    """The solver's keyword parameters with their defaults, overridden by the options of this run."""
    settings = {}
    try:
        for parameter in list(inspect.signature(solver).parameters.values())[3:]:   # after problem, rng, initial
            if parameter.default is not parameter.empty:
                settings[parameter.name] = parameter.default
    except (TypeError, ValueError):
        pass
    settings.update(options)
    return {key: _plain(value) for key, value in settings.items()}


def _label(solver):
    name = getattr(solver, "__name__", None) or getattr(getattr(solver, "func", None), "__name__", "Search")
    return name.replace("_", " ").capitalize()


def run_search(solver, problem, seed, initial=None, record=True, label=None, **options):
    """Run ``solver(problem, rng, initial, **options)`` and return the finished ``SearchLog``: best tour and cost,
    ``steps`` and ``evaluations`` used, events, frames for the animation.

    The solver receives a RecordedProblem, a random generator built from ``seed`` and a copy of the initial state
    (a random one when ``initial`` is None); it stops by itself, after at most ``max_steps`` steps or earlier by its
    own rule, and returns nothing. ``record=False`` disables the trace for timed benchmarks; ``options`` (for
    example ``max_steps``) are passed to the solver and stored as the run's settings.
    """
    rng = np.random.default_rng(seed)
    if initial is None:
        initial = problem.random_state(rng) if hasattr(problem, "random_state") else problem.random_tour(rng)
    start = np.array(initial, copy=True)
    log = SearchLog(problem, seed, label=label or _label(solver), record=record,
                    settings=_settings(solver, options))
    recorded = RecordedProblem(problem, log)
    solver(recorded, rng, start, **options)
    return recorded._finish()


def as_method(solver, **options):
    """A method for ``benchmark``: method(problem, seed, initial, record=False) -> finished SearchLog."""
    def method(problem, seed, initial, record=False):
        return run_search(solver, problem, seed, initial, record=record, **options)
    method.__name__ = _label(solver)
    return method


# The guided example of the player (Section 2.2): one starting tour, then hand-authored 2-opt reversals whose
# acceptance is scripted, so that every frame shows one thing to read. Nothing here searches.
_DEMO_START = [0, 16, 5, 19, 4, 15, 13, 11, 1, 20, 18, 9, 8, 2, 22, 14, 7, 17, 6, 23, 21, 3, 10, 12]   # Section 1, SEED = 7
_DEMO_STEPS = [
    # (i, j, accepted, note): reverse positions i..j of the current tour, evaluate the candidate, then take it or not.
    (9, 17, True, "The candidate costs less (Δ < 0): accepted. Current and best archived move together. "
                  "Select Candidate tour to see the proposal, then Current tour."),
    (11, 18, True, "Another improving reversal, accepted. The current curve and the best curve still coincide."),
    (3, 17, False, "Δ > 0: rejected. The current tour does not change, but the evaluation was counted (3 → 4)."),
    (7, 13, True, "Accepted: Δ < 0. Inspect a city to check its two edges before and after the move."),
    (4, 9, False, "Rejected: Δ > 0. Hill climbing never accepts such a move."),
    (10, 13, True, "Δ > 0 and yet accepted, as simulated annealing sometimes does: the current cost rises and the "
                   "best archived tour keeps the previous tour. Compare Current tour and Best archived tour."),
    (8, 17, True, "Accepted: Δ < 0, but the current cost stays above the best archived: the archive does not move."),
    (15, 21, True, "Accepted and better than the archive: the best archived tour is updated."),
    (17, 20, False, "Rejected: Δ > 0. The dashed best curve stays flat while nothing better is found."),
    (8, 15, True, "Accepted: Δ < 0, a new best."),
    (11, 15, False, "Rejected: Δ > 0. End of the example: 12 evaluations, 11 proposals, 7 accepted."),
]


def animation_demo(problem):
    """A guided example for the player, not a solver: the starting tour of Section 1, then eleven hand-authored
    2-opt proposals accepted or rejected by hand (seven accepted, one of them worse). Twelve frames, each with a
    note that says what to look at."""
    if problem.n != 24:
        raise ValueError("The demonstration is for the provided 24-city instance.")
    tour = np.array(_DEMO_START, dtype=int)
    log = SearchLog(problem, seed=7, label="Guided example", mode="demo",
                    settings={"origin": "Hand-authored 2-opt reversals with scripted decisions; no search algorithm",
                              "start": "The initial tour of Section 1 (random_tour, seed 7)"})
    cost = log.evaluate(tour)
    log.record(tour, cost, note="Starting tour: one evaluation. Current = best archived, no candidate yet. "
                                "Step with → and read this box at every frame.")
    for i, j, accepted, note in _DEMO_STEPS:
        candidate = tour.copy()
        candidate[i:j + 1] = candidate[i:j + 1][::-1]          # one 2-opt reversal, positions i..j included
        candidate_cost = log.evaluate(candidate)                # counts one evaluation, archives the best tour
        delta = candidate_cost - cost
        if accepted:
            tour, cost = candidate, candidate_cost
        log.steps += 1
        log.record(tour, cost, candidate=candidate, candidate_cost=candidate_cost,
                   accepted=accepted, delta=delta, note=note)
    return log.finish()
