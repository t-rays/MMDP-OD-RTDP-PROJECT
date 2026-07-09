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
        self.actions = []
        state = self.mdp.initial_state()
        self.trajectory.append(state)
        
        import random
        rng = random.Random(seed)
        
        step_count = 0
        self.success = False
        while step_count < self.max_steps and not self.mdp.is_terminal(state):
            # Select action
            action = self.planner.policy_action(state, tie_rng=rng)
            self.actions.append(action)
            # Step environment
            state = self.mdp.sample_next(state, action, rng)
            self.trajectory.append(state)
            step_count += 1
            
            if self.mdp.is_terminal(state):
                self.success = True
                break
                
        self.max_steps = len(self.trajectory) - 1
        self.fig, self.ax = None, None
        self.colors = ['#ff7675', '#74b9ff', '#00b894', '#e17055', '#0984e3', '#b2bec3']
        self.tree_html = widgets.HTML(value="")
        
    def draw_grid(self, ax):
        grid = self.mdp.instance.grid_map
        ax.clear()
        ax.set_xlim(-0.5, grid.width - 0.5)
        ax.set_ylim(-0.5, grid.height - 0.5)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Build 2D array for the grid and use imshow for efficient, seamless rendering
        import numpy as np
        import matplotlib.colors as mcolors
        
        grid_matrix = np.zeros((grid.height, grid.width))
        for (x, y) in grid.obstacles:
            grid_matrix[y, x] = 1.0
            
        cmap = mcolors.ListedColormap(['white', '#2d3436'])
        ax.imshow(grid_matrix, cmap=cmap, extent=[-0.5, grid.width-0.5, grid.height-0.5, -0.5], origin='upper', interpolation='nearest')
            
        # Draw goals
        for i, (gx, gy) in enumerate(self.mdp.instance.goals):
            color = self.colors[i % len(self.colors)]
            ax.add_patch(Rectangle((gx-0.5, gy-0.5), 1, 1, facecolor=color, alpha=0.3))
            ax.text(gx, gy, f"G{i}", ha='center', va='center', fontweight='bold')
            
    def render_step(self, step: int):
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            
        self.draw_grid(self.ax)
        
        if not self.trajectory:
            self.ax.text(self.mdp.instance.grid_map.width/2, self.mdp.instance.grid_map.height/2, 
                         "No trajectory data", ha='center', va='center')
            self.fig.canvas.draw()
            return
            
        state = self.trajectory[step]
        
        # Draw agents
        for i, pos in enumerate(state):
            if pos != self.mdp.instance.goals[i]: # If not at goal
                color = self.colors[i % len(self.colors)]
                self.ax.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8)) # Finished
                
        status_text = "Success" if self.success and step == self.max_steps else "Running"
        self.ax.set_title(f"Step {step} / {self.max_steps} | Status: {status_text}")
        self.fig.canvas.draw()
        
        # Update SVG trees
        from tree_visualizer import BranchingTreeVisualizer
        try:
            # If we are at the very last step, there is no "next" action, so show the previous action or None
            act = self.actions[step] if step < len(self.actions) else None
            svg_no, svg_od = BranchingTreeVisualizer.generate_trees_svg(self.mdp.n_agents, step, act)
            self.tree_html.value = f"<div style='display: flex; flex-direction: column; gap: 20px; align-items: center;'>{svg_no}{svg_od}</div>"
        except Exception as e:
            self.tree_html.value = f"<b style='color:red;'>Graphviz error: {e}</b><br>Make sure 'graphviz' is installed."
        
    def show_with_tree(self):
        if not self.trajectory:
            print("No trajectory found to visualize.")
            return
            
        # 1. Grid Visualizer Output
        grid_output = widgets.Output()
        
        with grid_output:
            self.render_step(0)
            display(self.fig)
            plt.close(self.fig) # Prevent duplicate inline plotting
            
        slider = widgets.IntSlider(min=0, max=self.max_steps, step=1, value=0, description='Step:')
        
        def on_change(change):
            if change['name'] == 'value':
                with grid_output:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    self.render_step(change['new'])
                    display(self.fig)
                    
        slider.observe(on_change)
        
        play = widgets.Play(min=0, max=self.max_steps, step=1, interval=400, show_repeat=False)
        widgets.jslink((play, 'value'), (slider, 'value'))
        
        # Inject CSS to fix missing FontAwesome icons in VSCode Jupyter
        # We use \FE0E (Variation Selector-15) to force text-rendering instead of Windows emojis
        icon_fix_css = widgets.HTML("""
        <style>
        .widget-play .fa-play:before { content: "\\25b6\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        .widget-play .fa-pause:before { content: "\\23f8\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        .widget-play .fa-stop:before { content: "\\23f9\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        </style>
        """)
        
        controls = widgets.HBox([icon_fix_css, play, slider], layout=widgets.Layout(align_items='center', justify_content='center', margin='10px 0px 0px 0px'))
        
        left_side = widgets.VBox([grid_output, controls], layout=widgets.Layout(align_items='center'))
            
        # Combine Side-by-Side
        main_layout = widgets.HBox([left_side, self.tree_html], layout=widgets.Layout(align_items='center', justify_content='space-around'))
        display(main_layout)


class DualTrajectoryVisualizer:
    def __init__(self, mdp, baseline_planner, od_planner, max_steps=100, seed=42):
        self.mdp = mdp
        self.max_steps_limit = max_steps
        
        # Run Baseline
        self.traj_base, self.actions_base, self.success_base = self._simulate(baseline_planner, seed)
        # Run OD
        self.traj_od, self.actions_od, self.success_od = self._simulate(od_planner, seed)
        
        self.max_steps = max(len(self.traj_base), len(self.traj_od)) - 1
        self.fig, (self.ax1, self.ax2) = None, (None, None)
        self.colors = ['#ff7675', '#74b9ff', '#00b894', '#e17055', '#0984e3', '#b2bec3']
        self.tree_html = widgets.HTML(value="")
        
    def _simulate(self, planner, seed):
        import random
        rng = random.Random(seed)
        state = self.mdp.initial_state()
        trajectory = [state]
        actions = []
        step_count = 0
        success = False
        
        while step_count < self.max_steps_limit:
            if self.mdp.is_terminal(state):
                success = True
                break
                
            action = planner.policy_action(state)
            if action is None:
                break
                
            actions.append(action)
            state = self.mdp.sample_next(state, action, rng)
            trajectory.append(state)
            step_count += 1
            
            if self.mdp.is_terminal(state):
                success = True
                break
                
        return trajectory, actions, success

    def draw_grid(self, ax):
        grid = self.mdp.instance.grid_map
        ax.clear()
        
        import numpy as np
        import matplotlib.colors as mcolors
        
        grid_matrix = np.zeros((grid.height, grid.width))
        for (x, y) in grid.obstacles:
            grid_matrix[y, x] = 1.0
            
        cmap = mcolors.ListedColormap(['white', '#2d3436'])
        ax.imshow(grid_matrix, cmap=cmap, extent=[-0.5, grid.width-0.5, grid.height-0.5, -0.5], origin='upper', interpolation='nearest')
        
        ax.set_xlim(-0.5, grid.width - 0.5)
        ax.set_ylim(grid.height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        for i, (gx, gy) in enumerate(self.mdp.instance.goals):
            color = self.colors[i % len(self.colors)]
            ax.add_patch(Rectangle((gx-0.5, gy-0.5), 1, 1, facecolor=color, alpha=0.3))
            ax.text(gx, gy, f"G{i}", ha='center', va='center', fontweight='bold')

    def render_step(self, step: int):
        if not self.fig:
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
        self.draw_grid(self.ax1)
        self.draw_grid(self.ax2)
        
        # Baseline
        s_idx1 = min(step, len(self.traj_base)-1)
        state1 = self.traj_base[s_idx1]
        for i, pos in enumerate(state1):
            if pos != self.mdp.instance.goals[i]:
                color = self.colors[i % len(self.colors)]
                self.ax1.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax1.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax1.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8))
        
        status_text1 = "Success" if self.success_base and s_idx1 == len(self.traj_base)-1 else ("Failed" if s_idx1 == len(self.traj_base)-1 else "Running")
        self.ax1.set_title(f"Baseline (Joint) | Step {s_idx1} | {status_text1}")
        
        # OD
        s_idx2 = min(step, len(self.traj_od)-1)
        state2 = self.traj_od[s_idx2]
        for i, pos in enumerate(state2):
            if pos != self.mdp.instance.goals[i]:
                color = self.colors[i % len(self.colors)]
                self.ax2.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax2.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax2.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8))
        
        status_text2 = "Success" if self.success_od and s_idx2 == len(self.traj_od)-1 else ("Failed" if s_idx2 == len(self.traj_od)-1 else "Running")
        self.ax2.set_title(f"Operator Decomposition | Step {s_idx2} | {status_text2}")
        
        self.fig.canvas.draw()
        
        from tree_visualizer import BranchingTreeVisualizer
        try:
            act = self.actions_od[s_idx2] if s_idx2 < len(self.actions_od) else None
            svg_no, svg_od = BranchingTreeVisualizer.generate_trees_svg(self.mdp.n_agents, s_idx2, act)
            self.tree_html.value = f"<div style='display: flex; flex-direction: column; gap: 20px; align-items: center;'>{svg_no}{svg_od}</div>"
        except Exception as e:
            self.tree_html.value = f"<b style='color:red;'>Graphviz error: {e}</b><br>Make sure 'graphviz' is installed."
        
    def show_with_tree(self):
        if not self.traj_base and not self.traj_od:
            print("No trajectory found to visualize.")
            return
            
        grid_output = widgets.Output()
        with grid_output:
            self.render_step(0)
            display(self.fig)
            plt.close(self.fig)
            
        slider = widgets.IntSlider(min=0, max=self.max_steps, step=1, value=0, description='Step:', layout=widgets.Layout(width='300px'))
        
        def on_change(change):
            if change['name'] == 'value':
                with grid_output:
                    from IPython.display import clear_output
                    clear_output(wait=True)
                    self.render_step(change['new'])
                    display(self.fig)
                    
        slider.observe(on_change)
        
        play = widgets.Play(min=0, max=self.max_steps, step=1, interval=400, show_repeat=False)
        widgets.jslink((play, 'value'), (slider, 'value'))
        
        icon_fix_css = widgets.HTML("""
        <style>
        .widget-play .fa-play:before { content: "\\25b6\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        .widget-play .fa-pause:before { content: "\\23f8\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        .widget-play .fa-stop:before { content: "\\23f9\\fe0e" !important; font-family: Arial, sans-serif !important; font-size: 16px !important; color: #444 !important; }
        </style>
        """)
        
        controls = widgets.HBox([icon_fix_css, play, slider], layout=widgets.Layout(align_items='center', justify_content='center', margin='10px 0px 0px 0px'))
        display(widgets.VBox([grid_output, controls, self.tree_html], layout=widgets.Layout(align_items='center')))
