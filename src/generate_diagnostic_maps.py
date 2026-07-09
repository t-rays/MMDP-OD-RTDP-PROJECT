from __future__ import annotations

"""Generate small deterministic MovingAI-format maps for targeted diagnostics.

These maps complement the official MovingAI benchmark maps.  They are not
performance benchmarks; each one isolates a coordination mechanism:

* crossing-9-9: four one-cell-wide arms sharing a central intersection.
* corridor-passing-11-7: a one-cell corridor with a central passing bay.
* bottleneck-13-9: two rooms connected by a one-cell doorway.

Run from the project root:

    python src/generate_diagnostic_maps.py
"""

from collections import deque
from pathlib import Path

Position = tuple[int, int]
Task = tuple[Position, Position]


def _free_cells(rows: tuple[str, ...]) -> set[Position]:
    return {
        (x, y)
        for y, row in enumerate(rows)
        for x, symbol in enumerate(row)
        if symbol == "."
    }


def _distance(rows: tuple[str, ...], start: Position, goal: Position) -> int:
    free = _free_cells(rows)
    if start not in free or goal not in free:
        raise ValueError(f"Blocked diagnostic endpoint: {start} -> {goal}")
    queue: deque[tuple[Position, int]] = deque([(start, 0)])
    visited = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == goal:
            return distance
        for nxt in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nxt in free and nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, distance + 1))
    raise ValueError(f"Unreachable diagnostic task: {start} -> {goal}")


def _write_map(folder: Path, name: str, rows: tuple[str, ...]) -> None:
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"Unequal row widths for {name}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.map").write_text(
        "\n".join(
            [
                "type octile",
                f"height {len(rows)}",
                f"width {width}",
                "map",
                *rows,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scenario(
    folder: Path,
    name: str,
    rows: tuple[str, ...],
    scenario_number: int,
    tasks: tuple[Task, ...],
) -> None:
    width = len(rows[0])
    height = len(rows)
    lines = ["version 1"]
    for bucket, (start, goal) in enumerate(tasks):
        distance = _distance(rows, start, goal)
        lines.append(
            "\t".join(
                map(
                    str,
                    (
                        bucket,
                        f"{name}.map",
                        width,
                        height,
                        start[0],
                        start[1],
                        goal[0],
                        goal[1],
                        float(distance),
                    ),
                )
            )
        )
    scen_dir = folder / "scen"
    scen_dir.mkdir(parents=True, exist_ok=True)
    (scen_dir / f"{name}-diagnostic-{scenario_number}.scen").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def generate(root: Path) -> None:
    definitions: dict[
        str, tuple[tuple[str, ...], tuple[tuple[Task, ...], ...]]
    ] = {
        "crossing-9-9": (
            (
                "@@@@.@@@@",
                "@@@@.@@@@",
                "@@@@.@@@@",
                "@@@@.@@@@",
                ".........",
                "@@@@.@@@@",
                "@@@@.@@@@",
                "@@@@.@@@@",
                "@@@@.@@@@",
            ),
            (
                (
                    ((0, 4), (8, 4)),
                    ((8, 4), (0, 4)),
                    ((4, 0), (4, 8)),
                    ((4, 8), (4, 0)),
                ),
                (
                    ((0, 4), (4, 0)),
                    ((4, 0), (8, 4)),
                    ((8, 4), (4, 8)),
                    ((4, 8), (0, 4)),
                ),
                (
                    ((1, 4), (7, 4)),
                    ((7, 4), (1, 4)),
                    ((4, 1), (4, 7)),
                    ((4, 7), (4, 1)),
                ),
            ),
        ),
        "corridor-passing-11-7": (
            (
                "@@@@@@@@@@@",
                "@@@@@@@@@@@",
                "@@@@...@@@@",
                "@.........@",
                "@@@@...@@@@",
                "@@@@@@@@@@@",
                "@@@@@@@@@@@",
            ),
            (
                (
                    ((1, 3), (9, 3)),
                    ((9, 3), (1, 3)),
                    ((2, 3), (8, 3)),
                    ((8, 3), (2, 3)),
                    ((5, 2), (5, 4)),
                ),
                (
                    ((1, 3), (8, 3)),
                    ((9, 3), (2, 3)),
                    ((2, 3), (9, 3)),
                    ((8, 3), (1, 3)),
                    ((4, 2), (6, 4)),
                ),
                (
                    ((1, 3), (6, 4)),
                    ((9, 3), (4, 2)),
                    ((4, 2), (9, 3)),
                    ((6, 4), (1, 3)),
                    ((5, 4), (5, 2)),
                ),
            ),
        ),
        "bottleneck-13-9": (
            (
                "@@@@@@@@@@@@@",
                "@.....@.....@",
                "@.....@.....@",
                "@.....@.....@",
                "@...........@",
                "@.....@.....@",
                "@.....@.....@",
                "@.....@.....@",
                "@@@@@@@@@@@@@",
            ),
            (
                (
                    ((1, 1), (11, 1)),
                    ((1, 4), (11, 4)),
                    ((1, 7), (11, 7)),
                    ((11, 2), (1, 2)),
                    ((11, 6), (1, 6)),
                ),
                (
                    ((2, 2), (10, 6)),
                    ((2, 6), (10, 2)),
                    ((1, 4), (11, 4)),
                    ((10, 1), (2, 7)),
                    ((10, 7), (2, 1)),
                ),
                (
                    ((1, 2), (11, 6)),
                    ((1, 6), (11, 2)),
                    ((2, 4), (10, 4)),
                    ((11, 1), (1, 7)),
                    ((11, 7), (1, 1)),
                ),
            ),
        ),
    }

    for name, (rows, scenarios) in definitions.items():
        folder = root / name
        _write_map(folder, name, rows)
        for number, tasks in enumerate(scenarios, start=1):
            _write_scenario(folder, name, rows, number, tasks)

    readme = root / "README.md"
    readme.write_text(
        "# Diagnostic maps\n\n"
        "These maps are generated by `src/generate_diagnostic_maps.py`.\n"
        "They isolate crossing, passing-bay, and bottleneck coordination.\n"
        "They are correctness/diagnostic instances, not replacements for the\n"
        "official MovingAI benchmark maps.\n",
        encoding="utf-8",
    )



if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    generate(project_root / "maps" / "diagnostic")
    print("Diagnostic maps generated under maps/diagnostic")
