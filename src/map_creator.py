from __future__ import annotations

"""
Grid-map and scenario loader for the multi-agent planning project.

Expected project structure:

final_project/
└── maps/
    └── room-64-64-16/
        ├── room-64-64-16.map
        ├── scen/
        │   ├── room-64-64-16-even-1.scen
        │   └── ...
        └── room-64-64-16.pdf   # ignored by this module

The scenario directory can have any name. Scenario files are discovered
recursively under the selected map folder.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import argparse
import re


Position = tuple[int, int]

# Symbols used by the benchmark .map format.
TRAVERSABLE_SYMBOLS = frozenset({".", "G", "S"})
BLOCKED_SYMBOLS = frozenset({"@", "O", "T", "W"})


class MapCreatorError(ValueError):
    """Raised when a map folder, map file, or scenario file is invalid."""


@dataclass(frozen=True)
class GridMap:
    """Static grid geometry loaded from a .map file."""

    name: str
    path: Path
    width: int
    height: int
    grid: tuple[str, ...]
    obstacles: frozenset[Position]
    free_cells: frozenset[Position]

    def is_free(self, position: Position) -> bool:
        return position in self.free_cells

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors4(self, position: Position) -> Iterator[Position]:
        """Yield legal four-directional neighbors."""
        x, y = position

        for next_position in (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ):
            if next_position in self.free_cells:
                yield next_position


@dataclass(frozen=True)
class ScenarioEntry:
    """One start-goal task read from a .scen file."""

    bucket: int
    map_name: str
    width: int
    height: int
    start: Position
    goal: Position
    reference_distance: float
    source_file: Path
    source_line: int


@dataclass(frozen=True)
class MapInstance:
    """A multi-agent problem created from one grid map and scenario file."""

    grid_map: GridMap
    scenario_file: Path
    starts: tuple[Position, ...]
    goals: tuple[Position, ...]
    tasks: tuple[ScenarioEntry, ...]

    @property
    def n_agents(self) -> int:
        return len(self.starts)

    def summary(self) -> str:
        return (
            f"Map: {self.grid_map.name}\n"
            f"Size: {self.grid_map.width} x {self.grid_map.height}\n"
            f"Free cells: {len(self.grid_map.free_cells):,}\n"
            f"Obstacles: {len(self.grid_map.obstacles):,}\n"
            f"Scenario: {self.scenario_file.name}\n"
            f"Agents: {self.n_agents}\n"
            f"Starts: {self.starts}\n"
            f"Goals: {self.goals}"
        )


def _read_nonempty_lines(path: Path) -> list[str]:
    try:
        return [
            line.rstrip("\n\r")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise MapCreatorError(f"Could not read {path}: {exc}") from exc


def load_map_file(map_path: str | Path) -> GridMap:
    """
    Parse one benchmark .map file.

    The format header contains ``type octile`` even though this project uses
    only four-directional movement. The header check validates the file format;
    movement rules are defined later by GridMMDP and GridMap.neighbors4.
    """
    path = Path(map_path).resolve()

    if path.suffix.lower() != ".map":
        raise MapCreatorError(
            f"Expected a .map file, received: {path.name}"
        )

    if not path.is_file():
        raise MapCreatorError(
            f"Map file does not exist: {path}"
        )

    lines = _read_nonempty_lines(path)

    if len(lines) < 5:
        raise MapCreatorError(
            f"Map file is too short: {path}"
        )

    try:
        map_type_key, map_type = lines[0].split(maxsplit=1)
        height_key, height_text = lines[1].split(maxsplit=1)
        width_key, width_text = lines[2].split(maxsplit=1)
    except ValueError as exc:
        raise MapCreatorError(
            f"Malformed map header in {path}"
        ) from exc

    if map_type_key.lower() != "type":
        raise MapCreatorError(
            f"Expected 'type' header in {path}"
        )

    if map_type.lower() != "octile":
        raise MapCreatorError(
            f"Unsupported map type {map_type!r} in {path}; "
            "expected 'octile'"
        )

    if (
        height_key.lower() != "height"
        or width_key.lower() != "width"
    ):
        raise MapCreatorError(
            f"Expected height/width headers in {path}"
        )

    if lines[3].strip().lower() != "map":
        raise MapCreatorError(
            f"Expected 'map' marker on line 4 in {path}"
        )

    try:
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise MapCreatorError(
            f"Width and height must be integers in {path}"
        ) from exc

    if width <= 0 or height <= 0:
        raise MapCreatorError(
            f"Width and height must be positive in {path}"
        )

    grid_rows = lines[4:]

    if len(grid_rows) != height:
        raise MapCreatorError(
            f"{path.name}: header says height={height}, "
            f"but the file contains {len(grid_rows)} map rows"
        )

    obstacles: set[Position] = set()
    free_cells: set[Position] = set()

    for y, row in enumerate(grid_rows):
        if len(row) != width:
            raise MapCreatorError(
                f"{path.name}: row {y} has length {len(row)}, "
                f"expected {width}"
            )

        for x, symbol in enumerate(row):
            position = (x, y)

            if symbol in TRAVERSABLE_SYMBOLS:
                free_cells.add(position)
            elif symbol in BLOCKED_SYMBOLS:
                obstacles.add(position)
            else:
                raise MapCreatorError(
                    f"{path.name}: unknown map symbol {symbol!r} "
                    f"at {position}"
                )

    if not free_cells:
        raise MapCreatorError(
            f"{path.name} contains no traversable cells"
        )

    return GridMap(
        name=path.stem,
        path=path,
        width=width,
        height=height,
        grid=tuple(grid_rows),
        obstacles=frozenset(obstacles),
        free_cells=frozenset(free_cells),
    )


def load_scenario_file(
    scenario_path: str | Path,
    *,
    expected_map: GridMap | None = None,
) -> list[ScenarioEntry]:
    """
    Parse one .scen file.

    When ``expected_map`` is supplied, every task is checked against the map's
    filename, dimensions, boundaries, and blocked cells.
    """
    path = Path(scenario_path).resolve()

    if path.suffix.lower() != ".scen":
        raise MapCreatorError(
            f"Expected a .scen file, received: {path.name}"
        )

    if not path.is_file():
        raise MapCreatorError(
            f"Scenario file does not exist: {path}"
        )

    lines = _read_nonempty_lines(path)

    if not lines or not lines[0].lower().startswith("version"):
        raise MapCreatorError(
            f"Missing scenario version header in {path}"
        )

    entries: list[ScenarioEntry] = []

    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()

        if len(fields) != 9:
            raise MapCreatorError(
                f"{path.name}, line {line_number}: expected 9 fields, "
                f"received {len(fields)}"
            )

        try:
            entry = ScenarioEntry(
                bucket=int(fields[0]),
                map_name=fields[1],
                width=int(fields[2]),
                height=int(fields[3]),
                start=(int(fields[4]), int(fields[5])),
                goal=(int(fields[6]), int(fields[7])),
                reference_distance=float(fields[8]),
                source_file=path,
                source_line=line_number,
            )
        except ValueError as exc:
            raise MapCreatorError(
                f"{path.name}, line {line_number}: invalid numeric value"
            ) from exc

        if expected_map is not None:
            _validate_scenario_entry(entry, expected_map)

        entries.append(entry)

    if not entries:
        raise MapCreatorError(
            f"No scenario tasks found in {path}"
        )

    return entries


def _validate_scenario_entry(
    entry: ScenarioEntry,
    grid_map: GridMap,
) -> None:
    expected_filename = grid_map.path.name

    if entry.map_name != expected_filename:
        raise MapCreatorError(
            f"{entry.source_file.name}, line {entry.source_line}: "
            f"scenario belongs to {entry.map_name!r}, but the loaded map is "
            f"{expected_filename!r}"
        )

    if (entry.width, entry.height) != (
        grid_map.width,
        grid_map.height,
    ):
        raise MapCreatorError(
            f"{entry.source_file.name}, line {entry.source_line}: "
            f"scenario dimensions are {entry.width}x{entry.height}, "
            f"but the map is {grid_map.width}x{grid_map.height}"
        )

    for label, position in (
        ("start", entry.start),
        ("goal", entry.goal),
    ):
        if not grid_map.in_bounds(position):
            raise MapCreatorError(
                f"{entry.source_file.name}, line {entry.source_line}: "
                f"{label} {position} is outside the map"
            )

        if not grid_map.is_free(position):
            raise MapCreatorError(
                f"{entry.source_file.name}, line {entry.source_line}: "
                f"{label} {position} is blocked"
            )


def discover_map_file(map_folder: str | Path) -> Path:
    """Find the single .map file directly inside a map folder."""
    folder = Path(map_folder).resolve()

    if not folder.is_dir():
        raise MapCreatorError(
            f"Map folder does not exist: {folder}"
        )

    map_files = sorted(folder.glob("*.map"))

    if not map_files:
        raise MapCreatorError(
            f"No .map file found directly inside {folder}"
        )

    if len(map_files) > 1:
        names = ", ".join(path.name for path in map_files)
        raise MapCreatorError(
            f"Expected one .map file in {folder}, found: {names}"
        )

    return map_files[0]


def discover_scenario_files(
    map_folder: str | Path,
) -> list[Path]:
    """Find .scen files recursively under the selected map folder."""
    folder = Path(map_folder).resolve()

    def natural_key(path: Path) -> list[object]:
        relative_path = str(path.relative_to(folder))
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", relative_path)
        ]

    files = sorted(
        folder.rglob("*.scen"),
        key=natural_key,
    )

    if not files:
        raise MapCreatorError(
            f"No .scen files found under {folder}"
        )

    return files


def _reachable_4way(
    grid_map: GridMap,
    start: Position,
    goal: Position,
) -> bool:
    """Check reachability under the project's four-directional movement."""
    if start == goal:
        return True

    queue: deque[Position] = deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()

        for neighbor in grid_map.neighbors4(current):
            if neighbor == goal:
                return True

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def select_agent_tasks(
    entries: Sequence[ScenarioEntry],
    grid_map: GridMap,
    n_agents: int,
    *,
    offset: int = 0,
    require_unique_starts: bool = True,
    require_unique_goals: bool = True,
    require_4way_reachability: bool = True,
) -> tuple[ScenarioEntry, ...]:
    """
    Select a deterministic valid group of tasks for the requested agents.

    Invalid, duplicate, or unreachable entries are skipped until enough tasks
    have been collected.
    """
    if n_agents <= 0:
        raise MapCreatorError(
            "n_agents must be positive"
        )

    if offset < 0:
        raise MapCreatorError(
            "offset cannot be negative"
        )

    if offset >= len(entries):
        raise MapCreatorError(
            f"offset={offset} is outside the scenario file "
            f"with {len(entries)} entries"
        )

    selected: list[ScenarioEntry] = []
    used_starts: set[Position] = set()
    used_goals: set[Position] = set()

    for entry in entries[offset:]:
        _validate_scenario_entry(
            entry,
            grid_map,
        )

        if (
            require_unique_starts
            and entry.start in used_starts
        ):
            continue

        if (
            require_unique_goals
            and entry.goal in used_goals
        ):
            continue

        if (
            require_4way_reachability
            and not _reachable_4way(
                grid_map,
                entry.start,
                entry.goal,
            )
        ):
            continue

        selected.append(entry)
        used_starts.add(entry.start)
        used_goals.add(entry.goal)

        if len(selected) == n_agents:
            return tuple(selected)

    raise MapCreatorError(
        f"Could select only {len(selected)} valid tasks, "
        f"but {n_agents} agents were requested"
    )


def create_map_instance(
    map_folder: str | Path,
    n_agents: int,
    *,
    scenario_file: str | Path | None = None,
    scenario_number: int = 1,
    task_offset: int = 0,
    require_4way_reachability: bool = True,
) -> MapInstance:
    """Create one multi-agent instance from a map folder."""
    folder = Path(map_folder).resolve()
    map_path = discover_map_file(folder)
    grid_map = load_map_file(map_path)

    if scenario_file is None:
        scenario_files = discover_scenario_files(folder)

        if not 1 <= scenario_number <= len(scenario_files):
            raise MapCreatorError(
                f"scenario_number must be between 1 and "
                f"{len(scenario_files)}"
            )

        selected_scenario_path = scenario_files[
            scenario_number - 1
        ]
    else:
        scenario_path = Path(scenario_file)
        selected_scenario_path = (
            scenario_path
            if scenario_path.is_absolute()
            else folder / scenario_path
        ).resolve()

    entries = load_scenario_file(
        selected_scenario_path,
        expected_map=grid_map,
    )

    tasks = select_agent_tasks(
        entries,
        grid_map,
        n_agents,
        offset=task_offset,
        require_4way_reachability=require_4way_reachability,
    )

    return MapInstance(
        grid_map=grid_map,
        scenario_file=selected_scenario_path,
        starts=tuple(task.start for task in tasks),
        goals=tuple(task.goal for task in tasks),
        tasks=tasks,
    )


def render_instance(
    instance: MapInstance,
    output_path: str | Path,
) -> Path:
    """
    Save a visual validation image.

    If ``output_path`` is a directory, the function creates a default filename
    inside it, such as ``room-64-64-16_preview.png``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "render_instance requires matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    output = Path(output_path).resolve()

    if output.exists() and output.is_dir():
        output = output / (
            f"{instance.grid_map.name}_preview.png"
        )
    elif not output.suffix:
        output = output / (
            f"{instance.grid_map.name}_preview.png"
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = [
        [
            0
            if (x, y) in instance.grid_map.free_cells
            else 1
            for x in range(instance.grid_map.width)
        ]
        for y in range(instance.grid_map.height)
    ]

    figure, axis = plt.subplots(
        figsize=(10, 7)
    )

    axis.imshow(
        image,
        cmap="gray_r",
        interpolation="nearest",
    )

    start_x = [position[0] for position in instance.starts]
    start_y = [position[1] for position in instance.starts]
    goal_x = [position[0] for position in instance.goals]
    goal_y = [position[1] for position in instance.goals]

    axis.scatter(
        start_x,
        start_y,
        marker="o",
        label="Starts",
    )

    axis.scatter(
        goal_x,
        goal_y,
        marker="x",
        label="Goals",
    )

    for agent_index, (start, goal) in enumerate(
        zip(instance.starts, instance.goals),
        start=1,
    ):
        axis.text(
            start[0],
            start[1],
            str(agent_index),
            fontsize=8,
        )
        axis.text(
            goal[0],
            goal[1],
            str(agent_index),
            fontsize=8,
        )

    axis.set_title(
        f"{instance.grid_map.name}: {instance.n_agents} agents"
    )
    axis.set_xlim(
        -0.5,
        instance.grid_map.width - 0.5,
    )
    axis.set_ylim(
        instance.grid_map.height - 0.5,
        -0.5,
    )
    axis.set_aspect("equal")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output,
        dpi=180,
    )
    plt.close(figure)

    return output


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create and validate a multi-agent instance from a map folder."
        )
    )
    parser.add_argument(
        "map_folder",
        type=Path,
        help="Folder containing one .map file and scenario files",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=2,
        help="Number of agent tasks to select (default: 2)",
    )
    parser.add_argument(
        "--scenario-number",
        type=int,
        default=1,
        help="One-based index of discovered .scen files (default: 1)",
    )
    parser.add_argument(
        "--task-offset",
        type=int,
        default=0,
        help="Skip this many scenario rows before selecting tasks",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional PNG path or output directory",
    )
    parser.add_argument(
        "--allow-non-four-way",
        action="store_true",
        help="Do not reject tasks unreachable with four-way movement",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    try:
        instance = create_map_instance(
            args.map_folder,
            args.agents,
            scenario_number=args.scenario_number,
            task_offset=args.task_offset,
            require_4way_reachability=(
                not args.allow_non_four_way
            ),
        )
    except MapCreatorError as exc:
        raise SystemExit(
            f"Map creation failed: {exc}"
        ) from exc

    print(instance.summary())

    if args.preview is not None:
        preview_path = render_instance(
            instance,
            args.preview,
        )
        print(
            f"Preview image saved to: {preview_path}"
        )


if __name__ == "__main__":
    main()
