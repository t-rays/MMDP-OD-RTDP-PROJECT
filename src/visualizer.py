import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import ipywidgets as widgets
from IPython.display import display
from typing import Optional, List, Tuple
from grid_mmdp import GridMMDP

def _generate_bellman_html(mdp, planner, state, action):
    if action is None or mdp.is_terminal(state):
        return "<div style='padding: 10px; color: #7f8c8d; font-style: italic; font-family: sans-serif;'>Terminal state reached. V(s) = 0.0</div>"
        
    def get_v(s):
        if hasattr(planner, 'real_state_value'):
            return planner.real_state_value(s)
        return planner.value(s)
        
    transitions = mdp.joint_transitions(state, action)
    
    terms = []
    expected_q = 0.0
    for next_state, prob in transitions:
        cost = mdp.transition_cost(state, action, next_state)
        v_next = get_v(next_state)
        expected_q += prob * (cost + v_next)
        terms.append(f"{prob:.2f} &times; [{cost:.1f} + {v_next:.2f}]")
        
    if len(terms) > 3:
        chunks = [" + ".join(terms[i:i+3]) for i in range(0, len(terms), 3)]
        sum_str = " + <br>&nbsp;&nbsp;&nbsp;&nbsp;".join(chunks)
        sum_str = f"= {sum_str}"
    else:
        sum_str = "= " + " + ".join(terms)
        
    act_str = ", ".join(action)
    
    html = f"""
    <div style="background-color: #fdfbf7; padding: 15px; border-radius: 8px; border: 1px solid #e1d8c1; margin-top: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); min-width: 400px; max-width: 650px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h4 style="margin-top: 0; margin-bottom: 12px; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 8px;">Live Bellman Equation</h4>
        <div style="font-size: 1.1em; margin-bottom: 12px; color: #34495e; font-family: 'Consolas', 'Courier New', monospace;">
            <i>Q(s, a)</i> = <b>&Sigma;</b> <i>P(s'|s,a)</i> [ <i>c(s,a,s')</i> + <i>V(s')</i> ]
        </div>
        <div style="font-size: 1.05em; margin-bottom: 15px; color: #c0392b; font-family: 'Consolas', 'Courier New', monospace; line-height: 1.5;">
            {sum_str}
        </div>
        <div style="font-size: 1.2em; font-weight: bold; color: #27ae60; border-top: 1px dashed #ccc; padding-top: 12px; font-family: 'Consolas', 'Courier New', monospace;">
            <i>Q</i>(s, <span style="color: #2980b9;">[{act_str}]</span>) = {expected_q:.3f}
        </div>
    </div>
    """
    return html


def _build_tree_decision_context(mdp, planner, state, chosen_action, next_state=None):
    decision = {}
    info = {}
    ACTIONS = ['up', 'down', 'right', 'left', 'stay']
    
    # 1. Joint Scores
    scores = []
    chosen_idx = -1
    best_idx = -1
    best_val = -float('inf')
    
    import itertools
    all_joint = list(itertools.product(ACTIONS, repeat=mdp.n_agents))
    for i, joint_a in enumerate(all_joint):
        val = planner.complete_joint_action_value(state, joint_a, count_metrics=False)
        scores.append(val)
        if val > best_val:
            best_val = val
            best_idx = i
        if chosen_action and tuple(joint_a) == tuple(chosen_action):
            chosen_idx = i
            
    decision['joint_scores'] = scores
    decision['chosen_idx'] = chosen_idx
    decision['joint_best_idx'] = best_idx
    
    # 2. OD Actions & Q Values
    od_actions = []
    if chosen_action:
        od_actions = [ACTIONS.index(a) for a in chosen_action]
    decision['od_actions'] = od_actions
    
    od_q = []
    reserved_targets = []
    
    if hasattr(planner, 'operator_value'):
        prefix = ()
        for i in range(mdp.n_agents):
            agent_q = []
            for a in ACTIONS:
                v = planner.operator_value((state, prefix), a, count_metrics=False)
                agent_q.append(v)
            od_q.append(agent_q)
            if chosen_action:
                prefix = prefix + (chosen_action[i],)
                curr_pos = state[i]
                # Fallback vectors
                vectors = {'up': (0, -1), 'down': (0, 1), 'right': (1, 0), 'left': (-1, 0), 'stay': (0, 0)}
                dx, dy = vectors[chosen_action[i]]
                reserved_targets.append(f"({curr_pos[0]+dx}, {curr_pos[1]+dy})")
    
    decision['od_q'] = od_q
    decision['reserved_targets'] = reserved_targets
    
    # 3. Info (Slip)
    executed = []
    slipped = []
    if next_state and chosen_action:
        vectors = {'up': (0, -1), 'down': (0, 1), 'right': (1, 0), 'left': (-1, 0), 'stay': (0, 0)}
        for i in range(mdp.n_agents):
            curr_pos = state[i]
            dx, dy = vectors[chosen_action[i]]
            intended = (curr_pos[0]+dx, curr_pos[1]+dy)
            actual = next_state[i]
            
            actual_action = 'stay'
            for a, (ax, ay) in vectors.items():
                if (curr_pos[0]+ax, curr_pos[1]+ay) == actual:
                    actual_action = a
                    break
            executed.append(ACTIONS.index(actual_action))
            slipped.append(actual != intended)
            
    info['executed'] = executed
    info['slipped'] = slipped
    
    return decision, info

class TrajectoryVisualizer:
    def __init__(self, mdp: GridMMDP, planner, max_steps: int = 50, seed: int = 42, show_trails: bool = True, show_heatmap: bool = False, heatmap_agent: int = 0, dynamic_projection: bool = False):
        self.mdp = mdp
        self.planner = planner
        self.max_steps = max_steps
        self.show_trails = show_trails
        self.show_heatmap = show_heatmap
        self.heatmap_agent = heatmap_agent
        self.dynamic_projection = dynamic_projection
        self.ax_heatmap = None
        
        self.heatmap_data = None
        if self.show_heatmap and hasattr(self.planner, 'heuristic') and self.planner.heuristic is not None:
            import numpy as np
            grid = self.mdp.instance.grid_map
            self.heatmap_data = np.full((grid.height, grid.width), np.nan)
            if 0 <= self.heatmap_agent < self.mdp.n_agents:
                dist_table = self.planner.heuristic.distance_tables[self.heatmap_agent]
                for (x, y), dist in dist_table.items():
                    self.heatmap_data[y, x] = dist
                    
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
        self.bellman_html = widgets.HTML(value="")
        self.tree_zoom = 1.0
        self._last_svgs = ("", "")
        
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
            
        if getattr(self, 'show_heatmap', False) and getattr(self, 'heatmap_data', None) is not None:
            import copy
            hm_cmap = copy.copy(plt.get_cmap('Blues_r'))
            hm_cmap.set_bad(alpha=0.0)
            ax.imshow(self.heatmap_data, cmap=hm_cmap, extent=[-0.5, grid.width-0.5, grid.height-0.5, -0.5], origin='upper', interpolation='nearest', alpha=0.5)

        # Draw goals
        for i, (gx, gy) in enumerate(self.mdp.instance.goals):
            color = self.colors[i % len(self.colors)]
            ax.add_patch(Rectangle((gx-0.5, gy-0.5), 1, 1, facecolor=color, alpha=0.3))
            ax.text(gx, gy, f"G{i}", ha='center', va='center', fontweight='bold')
            
    def render_step(self, step: int):
        if not self.fig:
            width = self.mdp.instance.grid_map.width
            height = self.mdp.instance.grid_map.height
            aspect_ratio = width / height
            
            if getattr(self, 'dynamic_projection', False):
                base_fig_width = min(9.0, max(4.0, width * 0.15))
                fig_height = base_fig_width / aspect_ratio
                self.fig, (self.ax, self.ax_heatmap) = plt.subplots(1, 2, figsize=(base_fig_width * 2, fig_height))
            else:
                fig_width = min(15.0, max(5.0, width * 0.2))
                fig_height = fig_width / aspect_ratio
                self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height))
            
        self.draw_grid(self.ax)
        
        if not self.trajectory:
            self.ax.text(self.mdp.instance.grid_map.width/2, self.mdp.instance.grid_map.height/2, 
                         "No trajectory data", ha='center', va='center')
            self.fig.canvas.draw()
            return
            
        state = self.trajectory[step]
        
        # Draw trails if enabled
        if getattr(self, 'show_trails', True) and step > 0:
            for i in range(self.mdp.n_agents):
                history = [s[i] for s in self.trajectory[:step+1]]
                color = self.colors[i % len(self.colors)]
                for j in range(1, len(history)):
                    alpha = 0.1 + 0.7 * (j / len(history))
                    self.ax.plot([history[j-1][0], history[j][0]], [history[j-1][1], history[j][1]], color=color, alpha=alpha, linewidth=3)
                    
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
        
        if getattr(self, 'dynamic_projection', False) and self.ax_heatmap:
            self.draw_grid(self.ax_heatmap)
            self.ax_heatmap.set_title(f"Dynamic Projection (Agent {self.heatmap_agent})")
            
            import numpy as np
            import copy
            
            grid = self.mdp.instance.grid_map
            dyn_heatmap = np.full((grid.height, grid.width), np.nan)
            
            if 0 <= self.heatmap_agent < self.mdp.n_agents:
                for x in range(grid.width):
                    for y in range(grid.height):
                        if (x, y) not in grid.obstacles:
                            s_list = list(state)
                            s_list[self.heatmap_agent] = (x, y)
                            s_prime = tuple(s_list)
                            
                            val = self.planner.real_state_value(s_prime) if hasattr(self.planner, 'real_state_value') else self.planner.value(s_prime)
                            dyn_heatmap[y, x] = val
                            
            hm_cmap = copy.copy(plt.get_cmap('RdYlGn'))
            hm_cmap.set_bad(color='#2d3436')
            
            img = self.ax_heatmap.imshow(dyn_heatmap, cmap=hm_cmap, extent=[-0.5, grid.width-0.5, grid.height-0.5, -0.5], origin='upper', interpolation='nearest', alpha=0.6)
            
            if not hasattr(self, 'cbar') or self.cbar is None:
                self.cbar = self.fig.colorbar(img, ax=self.ax_heatmap, fraction=0.046, pad=0.04)
                self.cbar.set_label("State Value (Q)")
            else:
                self.cbar.update_normal(img)
            
            for i, pos in enumerate(state):
                color = self.colors[i % len(self.colors)]
                if i == self.heatmap_agent:
                    self.ax_heatmap.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='white', alpha=0.7, linestyle='--'))
                    self.ax_heatmap.text(pos[0], pos[1], str(i), ha='center', va='center', color='black')
                else:
                    self.ax_heatmap.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                    self.ax_heatmap.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')

        self.fig.canvas.draw()
        
        # Update SVG trees
        from tree_visualizer import TreeVisualizer
        try:
            act = self.actions[step] if step < len(self.actions) else None
            next_state = self.trajectory[step+1] if step+1 < len(self.trajectory) else None
            decision, info = _build_tree_decision_context(self.mdp, self.planner, state, act, next_state)
            
            slip_rate = getattr(self.mdp.config, 'slip_to_stay_probability', 0.1)
            tv = TreeVisualizer(self.mdp.n_agents, slip_rate)
            svg_no, svg_od = tv.add_step(step, decision, info)
            self._last_svgs = (svg_no, svg_od)
            self.update_tree_html()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.tree_html.value = f"<b style='color:red;'>Graphviz error: {e}<br><pre>{err}</pre></b><br>Make sure 'graphviz' is installed."
            
        self.bellman_html.value = _generate_bellman_html(self.mdp, self.planner, state, act)
        

    def update_tree_html(self):
        svg_no, svg_od = getattr(self, '_last_svgs', ("", ""))
        zoom = getattr(self, 'tree_zoom', 1.0)
        self.tree_html.value = f"<div style='overflow: auto; max-height: 800px; width: 100%; border: 1px solid #ddd; padding: 20px; background: #fafafa;'><div style='display: flex; flex-direction: column; gap: 40px; align-items: center; transform: scale({zoom}); transform-origin: top center; transition: transform 0.1s ease-in-out;'>{svg_no}{svg_od}</div></div>"
    
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
        
        zoom_slider = widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description='Tree Zoom:')
        def on_zoom(change):
            self.tree_zoom = change['new']
            self.update_tree_html()
        zoom_slider.observe(on_zoom, names='value')
        
        controls = widgets.HBox([icon_fix_css, play, slider, zoom_slider], layout=widgets.Layout(align_items='center', justify_content='center', margin='10px 0px 0px 0px'))
        
        left_side = widgets.VBox([grid_output, controls, self.bellman_html], layout=widgets.Layout(align_items='center'))
            
        # Combine Side-by-Side
        main_layout = widgets.VBox([left_side, self.tree_html], layout=widgets.Layout(align_items='center'))
        display(main_layout)


class DualTrajectoryVisualizer:
    def __init__(self, mdp, baseline_planner, od_planner, max_steps=100, seed=42, show_trails=True, show_heatmap=False, heatmap_agent=0):
        self.mdp = mdp
        self.max_steps_limit = max_steps
        self.show_trails = show_trails
        self.show_heatmap = show_heatmap
        self.heatmap_agent = heatmap_agent
        
        self.heatmap_data = None
        if self.show_heatmap and hasattr(baseline_planner, 'heuristic') and baseline_planner.heuristic is not None:
            import numpy as np
            grid = self.mdp.instance.grid_map
            self.heatmap_data = np.full((grid.height, grid.width), np.nan)
            if 0 <= self.heatmap_agent < self.mdp.n_agents:
                dist_table = baseline_planner.heuristic.distance_tables[self.heatmap_agent]
                for (x, y), dist in dist_table.items():
                    self.heatmap_data[y, x] = dist
                    
        # Run Baseline
        self.traj_base, self.actions_base, self.success_base = self._simulate(baseline_planner, seed)
        # Run OD
        self.traj_od, self.actions_od, self.success_od = self._simulate(od_planner, seed)
        
        self.max_steps = max(len(self.traj_base), len(self.traj_od)) - 1
        self.fig, (self.ax1, self.ax2) = None, (None, None)
        self.colors = ['#ff7675', '#74b9ff', '#00b894', '#e17055', '#0984e3', '#b2bec3']
        self.tree_html = widgets.HTML(value="")
        self.bellman_base_html = widgets.HTML(value="")
        self.bellman_od_html = widgets.HTML(value="")
        self.tree_zoom = 1.0
        self._last_svgs = ("", "")
        self.baseline_planner = baseline_planner
        self.od_planner = od_planner
        
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
        
        if getattr(self, 'show_heatmap', False) and getattr(self, 'heatmap_data', None) is not None:
            import copy
            hm_cmap = copy.copy(plt.get_cmap('Blues_r'))
            hm_cmap.set_bad(alpha=0.0)
            ax.imshow(self.heatmap_data, cmap=hm_cmap, extent=[-0.5, grid.width-0.5, grid.height-0.5, -0.5], origin='upper', interpolation='nearest', alpha=0.5)

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
            width = self.mdp.instance.grid_map.width
            height = self.mdp.instance.grid_map.height
            aspect_ratio = width / height
            
            base_fig_width = min(9.0, max(4.0, width * 0.15))
            fig_height = base_fig_width / aspect_ratio
            
            self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(base_fig_width * 2, fig_height))
            
        self.draw_grid(self.ax1)
        self.draw_grid(self.ax2)
        
        # Baseline
        s_idx1 = min(step, len(self.traj_base)-1)
        
        if getattr(self, 'show_trails', True) and s_idx1 > 0:
            for i in range(self.mdp.n_agents):
                history = [s[i] for s in self.traj_base[:s_idx1+1]]
                color = self.colors[i % len(self.colors)]
                for j in range(1, len(history)):
                    alpha = 0.1 + 0.7 * (j / len(history))
                    self.ax1.plot([history[j-1][0], history[j][0]], [history[j-1][1], history[j][1]], color=color, alpha=alpha, linewidth=3)
                    
        state1 = self.traj_base[s_idx1]
        act1 = self.actions_base[s_idx1] if s_idx1 < len(self.actions_base) else None
        for i, pos in enumerate(state1):
            if pos != self.mdp.instance.goals[i]:
                color = self.colors[i % len(self.colors)]
                self.ax1.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax1.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax1.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8))
        
        status_text1 = "Success" if self.success_base and s_idx1 == len(self.traj_base)-1 else ("Failed" if s_idx1 == len(self.traj_base)-1 else "Running")
        self.ax1.set_title(f"Baseline (Joint) | Step {s_idx1} | {status_text1}")
        self.bellman_base_html.value = _generate_bellman_html(self.mdp, self.baseline_planner, state1, act1)
        
        # OD
        s_idx2 = min(step, len(self.traj_od)-1)
        
        if getattr(self, 'show_trails', True) and s_idx2 > 0:
            for i in range(self.mdp.n_agents):
                history = [s[i] for s in self.traj_od[:s_idx2+1]]
                color = self.colors[i % len(self.colors)]
                for j in range(1, len(history)):
                    alpha = 0.1 + 0.7 * (j / len(history))
                    self.ax2.plot([history[j-1][0], history[j][0]], [history[j-1][1], history[j][1]], color=color, alpha=alpha, linewidth=3)
                    
        state2 = self.traj_od[s_idx2]
        act2 = self.actions_od[s_idx2] if s_idx2 < len(self.actions_od) else None
        for i, pos in enumerate(state2):
            if pos != self.mdp.instance.goals[i]:
                color = self.colors[i % len(self.colors)]
                self.ax2.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor=color))
                self.ax2.text(pos[0], pos[1], str(i), ha='center', va='center', color='white', fontweight='bold')
            else:
                self.ax2.add_patch(Circle((pos[0], pos[1]), 0.4, facecolor='#00b894', alpha=0.8))
        
        status_text2 = "Success" if self.success_od and s_idx2 == len(self.traj_od)-1 else ("Failed" if s_idx2 == len(self.traj_od)-1 else "Running")
        self.ax2.set_title(f"Operator Decomposition | Step {s_idx2} | {status_text2}")
        self.bellman_od_html.value = _generate_bellman_html(self.mdp, self.od_planner, state2, act2)
        
        self.fig.canvas.draw()
        
        from tree_visualizer import TreeVisualizer
        try:
            act = self.actions_od[s_idx2] if s_idx2 < len(self.actions_od) else None
            next_state = self.traj_od[s_idx2+1] if s_idx2+1 < len(self.traj_od) else None
            decision, info = _build_tree_decision_context(self.mdp, self.od_planner, state2, act, next_state)
            
            slip_rate = getattr(self.mdp.config, 'slip_to_stay_probability', 0.1)
            tv = TreeVisualizer(self.mdp.n_agents, slip_rate)
            svg_no, svg_od = tv.add_step(s_idx2, decision, info)
            self._last_svgs = (svg_no, svg_od)
            self.update_tree_html()
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self.tree_html.value = f"<b style='color:red;'>Graphviz error: {e}<br><pre>{err}</pre></b><br>Make sure 'graphviz' is installed."
        

    def update_tree_html(self):
        svg_no, svg_od = getattr(self, '_last_svgs', ("", ""))
        zoom = getattr(self, 'tree_zoom', 1.0)
        self.tree_html.value = f"<div style='overflow: auto; max-height: 800px; width: 100%; border: 1px solid #ddd; padding: 20px; background: #fafafa;'><div style='display: flex; flex-direction: column; gap: 40px; align-items: center; transform: scale({zoom}); transform-origin: top center; transition: transform 0.1s ease-in-out;'>{svg_no}{svg_od}</div></div>"
    
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
        
        zoom_slider = widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description='Tree Zoom:')
        def on_zoom(change):
            self.tree_zoom = change['new']
            self.update_tree_html()
        zoom_slider.observe(on_zoom, names='value')
        
        controls = widgets.HBox([icon_fix_css, play, slider, zoom_slider], layout=widgets.Layout(align_items='center', justify_content='center', margin='10px 0px 0px 0px'))
        bellmans = widgets.HBox([self.bellman_base_html, self.bellman_od_html], layout=widgets.Layout(justify_content='space-around', width='100%'))
        display(widgets.VBox([grid_output, controls, bellmans, self.tree_html], layout=widgets.Layout(align_items='center')))
