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
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    for i, (start, goal) in enumerate(zip(starts, goals)):
        c = colors[i]
        
        # Draw a faint dashed line connecting Start and Goal for instant visual clarity
        ax.plot([start[0], goal[0]], [start[1], goal[1]], color=c, linestyle='--', alpha=0.5, linewidth=2)
        
        # Draw Start: Solid Circle with Agent ID
        ax.plot(start[0], start[1], marker='o', markersize=22, markerfacecolor=c, markeredgecolor='black', markeredgewidth=1.5)
        ax.text(start[0], start[1], f'{i+1}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        
        # Draw Goal: Solid Star with Agent ID
        ax.plot(goal[0], goal[1], marker='*', markersize=30, markerfacecolor=c, markeredgecolor='black', markeredgewidth=1.5)
        ax.text(goal[0], goal[1], f'{i+1}', ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        
    ax.set_title(f'Map Layout: {map_name} ({num_agents} Agents)', fontsize=16, pad=15)
    ax.set_xticks(np.arange(-0.5, grid_map.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_map.height, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()
