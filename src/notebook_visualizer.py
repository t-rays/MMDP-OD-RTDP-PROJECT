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
    
    img = np.ones((grid_map.height, grid_map.width, 3))
    for (x, y) in grid_map.obstacles:
        img[y, x] = [0.2, 0.2, 0.2] # Dark grey obstacles
        
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i)[:3] for i in range(num_agents)]
    
    for i, start in enumerate(starts):
        img[start[1], start[0]] = colors[i]
    for i, goal in enumerate(goals):
        img[goal[1], goal[0]] = [c*0.6 for c in colors[i]]
        
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    
    for i, start in enumerate(starts):
        ax.plot(start[0], start[1], marker='o', markersize=20, markerfacecolor='white', markeredgecolor='black')
        ax.text(start[0], start[1], f'{i+1}', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
    for i, goal in enumerate(goals):
        ax.plot(goal[0], goal[1], marker='*', markersize=28, markerfacecolor='white', markeredgecolor='black')
        ax.text(goal[0], goal[1], f'{i+1}', ha='center', va='center', color='black', fontweight='bold', fontsize=10)
        
    ax.set_title(f'Visualizing Map: {map_name} ({num_agents} Agents)', fontsize=14)
    ax.set_xticks(np.arange(-0.5, grid_map.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.height, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
