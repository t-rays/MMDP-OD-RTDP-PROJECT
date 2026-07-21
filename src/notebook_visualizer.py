import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Import the existing map creator logic
from map_creator import load_map_file, load_scenario_file

def plot_map_visualization(map_name, scen_name, num_agents):
    map_path = PROJECT_ROOT / 'maps' / map_name / f'{map_name}.map'
    scen_path = PROJECT_ROOT / 'maps' / map_name / 'scen' / f'{scen_name}.scen'
    
    grid_map = load_map_file(map_path)
    scenarios = load_scenario_file(scen_path, expected_map=grid_map)
    
    tasks = scenarios[:num_agents]
    starts = [t.start for t in tasks]
    goals = [t.goal for t in tasks]
    
    # We only color obstacles, floors remain white (img is initialized to 1s)
    img = np.ones((grid_map.height, grid_map.width, 3))
    for (x, y) in grid_map.obstacles:
        img[y, x] = [0.3, 0.3, 0.3] # Dark grey obstacles
        
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i)[:3] for i in range(num_agents)]
    
    # Dynamically scale figure and marker sizes based on grid dimensions
    max_dim = max(grid_map.width, grid_map.height)
    scale = 10.0 / max_dim  # Calibrated for an 8x8 map to look perfect
    
    # Calculate proportional figure size
    fig_width = min(12.0, max(6.0, 10.0 * (grid_map.width / max_dim)))
    fig_height = min(12.0, max(6.0, 10.0 * (grid_map.height / max_dim)))
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(img)
    
    # Dynamic marker sizes based on scale
    start_ms = max(4, int(22 * scale))
    goal_ms = max(6, int(30 * scale))
    f_size = max(5, int(12 * scale))
    l_width = max(0.5, 2 * scale)
    border_w = max(0.2, 1.5 * scale)
    
    for i, (start, goal) in enumerate(zip(starts, goals)):
        c = colors[i]
        
        # Draw a faint dashed line connecting Start and Goal for instant visual clarity
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color=c, linestyle='--', alpha=0.5, linewidth=l_width)
        
        # Draw Start: Solid Circle with Agent ID
        ax.plot(start[0], start[1], marker='o', markersize=start_ms, markerfacecolor=c, markeredgecolor='black', markeredgewidth=border_w)
        
        # Draw Goal: Solid Star with Agent ID
        ax.plot(goal[0], goal[1], marker='*', markersize=goal_ms, markerfacecolor=c, markeredgecolor='black', markeredgewidth=border_w)
        
        # Only draw text if it's large enough to be readable, otherwise leave it blank to avoid clutter
        if f_size >= 6:
            ax.text(start[0], start[1], f'{i+1}', ha='center', va='center', color='white', fontweight='bold', fontsize=f_size)
            ax.text(goal[0], goal[1], f'{i+1}', ha='center', va='center', color='white', fontweight='bold', fontsize=f_size)
        
    ax.set_title(f'Map Layout: {map_name} ({num_agents} Agents)', fontsize=16, pad=15)
    ax.set_xticks(np.arange(-0.5, grid_map.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.height, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
