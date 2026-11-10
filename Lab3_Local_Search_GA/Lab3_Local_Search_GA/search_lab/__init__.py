"""Technical helpers for the local-search practical: data, checks, recording, experiments, figures.

The notebook supplies the TSP model, the Sudoku model and every search algorithm; nothing here searches.
`run_search` runs an algorithm, which stops by itself after at most `max_steps` steps, and records its trace without
any logging code in the algorithm: the problem it hands to the algorithm counts `cost` calls and listens to
`neighbors`, `random_neighbor` and `snapshot`."""
from .data import FRENCH_CITIES
from .checks import check_neighbor, check_neighborhood, check_crossover, tour_report, cycle_diversity
from .recording import SearchLog, RecordedProblem, run_search, as_method, animation_demo
from .experiments import benchmark, summarize, paired_effect, export_results, environment
from .visuals import (show_tour, show_network, show_move, plot_trace,
                      animate_run, plot_results, plot_schedules, save_player)
from .sudoku import (SUDOKU_PUZZLES, SUDOKU_SOLUTIONS, parse_sudoku, check_sudoku_model, check_sudoku_neighbor,
                     check_sudoku_crossover, solve_summary, show_sudoku, plot_sudoku_trace, animate_sudoku,
                     save_sudoku_player)

__version__ = "4.3.0"
