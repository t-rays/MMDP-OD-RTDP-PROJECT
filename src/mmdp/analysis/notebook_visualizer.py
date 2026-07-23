"""Map and start-goal visualization used by the Colab notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mmdp.domain.map_creator import load_map_file, load_scenario_file


def _default_maps_root() -> Path:
    local_maps = Path.cwd() / "maps"
    if local_maps.is_dir():
        return local_maps
    return Path(__file__).resolve().parents[3] / "maps"


def plot_map_visualization(
    map_name: str,
    scenario_name: str,
    num_agents: int,
    maps_root: str | Path | None = None,
) -> None:
    """Display one benchmark map and the selected start-goal pairs."""
    root = Path(maps_root) if maps_root is not None else _default_maps_root()
    grid_map = load_map_file(root / map_name / f"{map_name}.map")
    scenarios = load_scenario_file(
        root / map_name / "scen" / f"{scenario_name}.scen",
        expected_map=grid_map,
    )
    tasks = scenarios[:num_agents]

    image = np.ones((grid_map.height, grid_map.width, 3))
    for x, y in grid_map.obstacles:
        image[y, x] = [0.3, 0.3, 0.3]

    max_dimension = max(grid_map.width, grid_map.height)
    scale = 10.0 / max_dimension
    figure_width = min(12.0, max(6.0, 10.0 * grid_map.width / max_dimension))
    figure_height = min(12.0, max(6.0, 10.0 * grid_map.height / max_dimension))
    colors = [plt.get_cmap("tab10")(index)[:3] for index in range(num_agents)]

    figure, axis = plt.subplots(figsize=(figure_width, figure_height))
    axis.imshow(image)

    start_size = max(4, int(22 * scale))
    goal_size = max(6, int(30 * scale))
    font_size = max(5, int(12 * scale))
    line_width = max(0.5, 2 * scale)
    border_width = max(0.2, 1.5 * scale)

    for index, task in enumerate(tasks):
        color = colors[index]
        start, goal = task.start, task.goal
        axis.plot(
            [start[0], goal[0]],
            [start[1], goal[1]],
            color=color,
            linestyle="--",
            alpha=0.5,
            linewidth=line_width,
        )
        axis.plot(
            start[0],
            start[1],
            marker="o",
            markersize=start_size,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=border_width,
        )
        axis.plot(
            goal[0],
            goal[1],
            marker="*",
            markersize=goal_size,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=border_width,
        )
        if font_size >= 6:
            label = str(index + 1)
            for position in (start, goal):
                axis.text(
                    position[0],
                    position[1],
                    label,
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=font_size,
                )

    axis.set_title(f"Map layout: {map_name} ({num_agents} agents)", fontsize=16, pad=15)
    axis.set_xticks(np.arange(-0.5, grid_map.width, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid_map.height, 1), minor=True)
    axis.grid(which="minor", color="black", linestyle="-", linewidth=1)
    axis.tick_params(which="minor", size=0)
    axis.set_xticks([])
    axis.set_yticks([])
    plt.show()
