import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import ipywidgets as widgets
from IPython.display import display
from typing import Optional, List, Tuple
from grid_mmdp import GridMMDP

class TrajectoryVisualizer:
    def __init__(self, mdp: GridMMDP, planner, max_steps: int = 50, seed: int = 42):
        self.mdp = mdp
        self.planner = planner
        self.max_steps = max_steps
        
        # Generate the trajectory manually
        self.trajectory = []
        state = self.mdp.initial_state()
        self.trajectory.append(state)
        
        import random
        rng = random.Random(seed)
        
        step_count = 0
        self.success = False
        while step_count < self.max_steps and not self.mdp.is_terminal(state):
            # Select action
            action = self.planner.policy_action(state, tie_rng=rng)
            # Step environment
            state, _ = self.mdp.sample_next(state, action, rng)
            self.trajectory.append(state)
            step_count += 1
            
            if self.mdp.is_terminal(state):
                self.success = True
                break
                
        self.max_steps = len(self.trajectory) - 1
        self.fig, self.ax = None, None
        self.colors = ['#ff7675', '#74b9ff', '#00b894', '#e17055', '#0984e3', '#b2bec3']
        
    def draw_grid(self, ax):
        grid = self.mdp.map_instance.grid_map
        ax.clear()
        ax.set_xlim(-0.5, grid.width - 0.5)
        ax.set_ylim(-0.5, grid.height - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Draw obstacles
        for (x, y) in grid.obstacles:
            ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, facecolor='#2d3436'))
            
        # Draw goals
        for i, (gx, gy) in enumerate(self.mdp.map_instance.goals):
            color = self.colors[i % len(self.colors)]
            ax.add_patch(Rectangle((gx-0.5, gy-0.5), 1, 1, facecolor=color, alpha=0.3))
            ax.text(gx, gy, f"G{i}", ha='center', va='center', fontweight='bold')
            
    def render_step(self, step: int):
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            
        self.draw_grid(self.ax)
        
        if not self.trajectory:
            self.ax.text(self.mdp.map_instance.grid_map.width/2, self.mdp.map_instance.grid_map.height/2, 
                         "No trajectory data", ha='center', va='center')
            self.fig.canvas.draw()
            return
            
        state = self.trajectory[step]
        
        # Draw agents
        for i, pos in enumerate(state.agent_positions):
            if pos != self.mdp.map_instance.goals[i]: # If not at goal
                color = self.colors[i % len(self.colors)]
                self.ax.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8)) # Finished
                
        status_text = "Success" if self.success and step == self.max_steps else "Running"
        self.ax.set_title(f"Step {step} / {self.max_steps} | Status: {status_text}")
        self.fig.canvas.draw()
        
    def show(self):
        if not self.trajectory:
            print("No trajectory found to visualize.")
            return
            
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        slider = widgets.IntSlider(min=0, max=self.max_steps, step=1, value=0, description='Step:')
        
        def on_change(change):
            if change['name'] == 'value':
                self.render_step(change['new'])
                
        slider.observe(on_change)
        self.render_step(0)
        
        # Controls
        play = widgets.Play(min=0, max=self.max_steps, step=1, interval=400, description="Press play")
        widgets.jslink((play, 'value'), (slider, 'value'))
        
        controls = widgets.HBox([play, slider])
        display(controls)
        plt.show()
