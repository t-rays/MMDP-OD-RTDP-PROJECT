from __future__ import annotations

"""Command-line entry point for paired Baseline RTDP / OD-RTDP experiments.

All experiment logic lives in the installed ``mmdp`` package
(``mmdp.experiments``); this script only parses and validates arguments.
Install the package first with ``pip install -e .`` from the repository root.
"""

import argparse
from pathlib import Path

from mmdp.experiments.factory import ALGORITHMS
from mmdp.experiments.profiles import RESOURCE_MODES
from mmdp.experiments.runner import run_experiments
from mmdp.analysis.statistics_utils import (
    binomial_worst_case_sample_size,
    consecutive_trials_for_detection,
)


def parse_optional_int(text: str) -> int | None:
    """Parse an integer or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected an integer or 'none', received {text!r}"
        ) from exc

    if value <= 0:
        raise argparse.ArgumentTypeError("The value must be positive or 'none'")

    return value


def parse_optional_nonnegative_int(text: str) -> int | None:
    """Parse a non-negative integer or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer or 'none', received {text!r}"
        ) from exc

    if value < 0:
        raise argparse.ArgumentTypeError("The value must be non-negative or 'none'")

    return value


def parse_optional_float(text: str) -> float | None:
    """Parse a positive float or the word 'none'."""
    normalized = text.strip().lower()

    if normalized in {"none", "null", "off", "disabled"}:
        return None

    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a number or 'none', received {text!r}"
        ) from exc

    if value <= 0.0:
        raise argparse.ArgumentTypeError("The value must be positive or 'none'")

    return value


def parse_optional_nonnegative_float(text: str) -> float | None:
    normalized = text.strip().lower()
    if normalized in {"none", "null", "off", "disabled"}:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative number or 'none', received {text!r}"
        ) from exc
    if value < 0.0:
        raise argparse.ArgumentTypeError("The value must be non-negative or 'none'")
    return value


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Baseline RTDP and OD-RTDP experiments with optional "
            "time and process-memory resource regimes."
        )
    )
    parser.add_argument("map_folders", type=Path, nargs="+")
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--algorithms", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS)
    )

    parser.add_argument("--planning-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--master-seed", type=int, default=20260708)

    parser.add_argument("--scenario-numbers", type=int, nargs="+", default=[1])
    parser.add_argument("--task-offsets", type=int, nargs="+", default=[0])
    parser.add_argument("--slip", type=float, default=0.20)

    parser.add_argument(
        "--resource-mode",
        choices=RESOURCE_MODES,
        default="custom",
        help="custom, unconstrained, time, memory, or time_memory",
    )
    parser.add_argument("--resource-profile", type=Path)
    parser.add_argument("--memory-limit-mb", type=parse_optional_float, default=None)
    parser.add_argument("--time-limit-seconds", type=parse_optional_float, default=None)
    parser.add_argument("--max-trials", type=parse_optional_int, default=None)
    parser.add_argument("--max-steps-per-trial", type=parse_optional_int, default=None)
    parser.add_argument(
        "--step-limit-multiplier",
        type=parse_optional_float,
        default=None,
        help="Legacy override. Default uses the stochastic tail bound.",
    )
    parser.add_argument(
        "--step-tail-probability", type=parse_optional_float, default=None,
        help="Explicit per-agent tail probability. Default derives it from the family-wise error target.",
    )
    parser.add_argument(
        "--step-cap-familywise-error", type=float, default=0.01,
        help="Target upper bound across all evaluation episodes and agents.",
    )

    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--relative-epsilon", type=float, default=1e-6)
    parser.add_argument("--stable-trials-required", type=parse_optional_int, default=None)
    parser.add_argument("--stability-confidence", type=float, default=0.99)
    parser.add_argument("--minimum-unstable-trial-rate", type=float, default=0.10)
    parser.add_argument("--stop-when-stable", action="store_true")
    parser.add_argument(
        "--stop-when-solved",
        action="store_true",
        help=(
            "Use LRTDP-style solved-state stopping. This is selected "
            "automatically by unconstrained, memory, time_or_solved, "
            "and time_memory_or_solved resource modes."
        ),
    )
    parser.add_argument(
        "--transition-cache-max-entries",
        type=parse_optional_nonnegative_int,
        default=None,
        help="Optional cache-entry proxy limit. Final memory modes should use RSS.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=parse_optional_nonnegative_float,
        default=None,
        help="Explicit absolute tie tolerance; default uses ULP comparison.",
    )
    parser.add_argument("--tie-ulps", type=int, default=8)

    parser.add_argument("--evaluation-episodes", type=parse_optional_int, default=None)
    parser.add_argument("--evaluation-confidence", type=float, default=0.95)
    parser.add_argument("--evaluation-half-width", type=float, default=0.10)
    parser.add_argument("--evaluation-max-steps", type=parse_optional_int, default=None)
    parser.add_argument(
        "--evaluation-time-limit-seconds",
        type=parse_optional_float,
        default=None,
        help="Stop starting new evaluation episodes after this budget.",
    )
    parser.add_argument("--disable-conflict-risk", action="store_true")
    parser.add_argument("--randomize-evaluation-ties", action="store_true")
    parser.add_argument("--cache-all-evaluation-transitions", action="store_true")
    parser.add_argument("--disable-evaluation-diagnostics", action="store_true")
    parser.add_argument("--evaluate-od-global-diagnostic", action="store_true")
    parser.add_argument("--diagnostics-output-dir", type=Path)

    parser.add_argument(
        "--output", type=Path, default=Path("results/raw_results.csv")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _resolve_derived_defaults(args: argparse.Namespace) -> None:
    if args.evaluation_episodes is None:
        args.evaluation_episodes = binomial_worst_case_sample_size(
            confidence=args.evaluation_confidence,
            half_width=args.evaluation_half_width,
        )
    if args.stable_trials_required is None:
        args.stable_trials_required = consecutive_trials_for_detection(
            confidence=args.stability_confidence,
            minimum_event_probability=args.minimum_unstable_trial_rate,
        )


def _validate_cli_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if any(count <= 0 for count in args.agent_counts):
        parser.error("Every agent count must be positive")
    if any(number <= 0 for number in args.scenario_numbers):
        parser.error("Scenario numbers must be positive")
    if any(offset < 0 for offset in args.task_offsets):
        parser.error("Task offsets cannot be negative")
    if not 0.0 <= args.slip < 1.0:
        parser.error("--slip must be in [0, 1)")
    if not 0.0 < args.evaluation_confidence < 1.0:
        parser.error("--evaluation-confidence must be in (0, 1)")
    if not 0.0 < args.evaluation_half_width < 1.0:
        parser.error("--evaluation-half-width must be in (0, 1)")
    if not 0.0 < args.stability_confidence < 1.0:
        parser.error("--stability-confidence must be in (0, 1)")
    if not 0.0 < args.minimum_unstable_trial_rate < 1.0:
        parser.error("--minimum-unstable-trial-rate must be in (0, 1)")
    if args.seed_count <= 0:
        parser.error("--seed-count must be positive")
    if args.evaluation_seeds is not None and args.planning_seeds is None:
        parser.error("Explicit evaluation seeds require explicit planning seeds")
    if args.epsilon < 0.0 or args.relative_epsilon < 0.0:
        parser.error("Residual tolerances cannot be negative")
    if (
        args.step_tail_probability is not None
        and not 0.0 < args.step_tail_probability < 1.0
    ):
        parser.error("--step-tail-probability must be in (0, 1) or none")
    if not 0.0 < args.step_cap_familywise_error < 1.0:
        parser.error("--step-cap-familywise-error must be in (0, 1)")
    if args.stable_trials_required is not None and args.stable_trials_required <= 0:
        parser.error("--stable-trials-required must be positive")
    if args.tie_ulps <= 0:
        parser.error("--tie-ulps must be positive")
    if args.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes must be positive")
    if (
        args.resource_mode == "custom"
        and args.max_trials is None
        and args.time_limit_seconds is None
        and args.memory_limit_mb is None
        and not args.stop_when_stable
        and not args.stop_when_solved
    ):
        parser.error("Custom mode needs a stopping mechanism")


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    _resolve_derived_defaults(args)
    _validate_cli_arguments(parser, args)
    run_experiments(args)


if __name__ == "__main__":
    main()
