import ipywidgets as widgets
from IPython.display import display
from pathlib import Path
import numpy as np

from grid_mmdp import GridMMDP, MMDPConfig
from map_creator import GridMap, MapInstance, ScenarioEntry
from heuristic import ShortestPathHeuristic
from od_rtdp import OperatorDecompositionRTDP
from baseline_rtdp import BaselineRTDP, RTDPConfig
from evaluation import evaluate_policy, EvaluationConfig
from resource_monitor import ResourceMonitor
from visualizer import TrajectoryVisualizer

class InteractiveSandbox:
    def __init__(self, initial_grid_size: int = 8, max_agents: int = 4):
        self.grid_size = initial_grid_size
        self.num_agents = 2
        self.max_agents = max_agents
        
        # Grid state: (x, y) -> string tool id
        # 'O' = Obstacle, 'F' = Free, 'S1' = Start 1, 'G1' = Goal 1, etc.
        self.grid_state = {} 
        self.buttons = {}
        
        # Colors
        self.colors = {
            'O': '#2d3436', # Dark Grey
            'F': '#dfe6e9', # Light Grey
            'S0': '#ff7675', 'G0': '#ff7675',
            'S1': '#74b9ff', 'G1': '#74b9ff',
            'S2': '#00b894', 'G2': '#00b894',
            'S3': '#e17055', 'G3': '#e17055',
            'S4': '#0984e3', 'G4': '#0984e3',
        }
        
        self._build_ui()
        self._initialize_grid()
        
    def _build_ui(self):
        # Controls
        self.agent_slider = widgets.IntSlider(min=1, max=self.max_agents, value=self.num_agents, description="Agents:")
        self.agent_slider.observe(self._on_agents_changed, names='value')
        
        self.run_button = widgets.Button(description="Run OD-RTDP", button_style="success", icon="play")
        self.run_button.on_click(self._on_run_clicked)
        
        self.status_label = widgets.Label(value="Status: Ready")
        
        controls = widgets.HBox([self.agent_slider, self.run_button, self.status_label])
        
        # Palette
        self.palette = widgets.ToggleButtons(
            options=self._get_palette_options(),
            description='Tool:',
            button_style='',
            tooltips=['Draw Obstacles', 'Erase to Free Space'] + ['Agent Start', 'Agent Goal'] * self.max_agents
        )
        
        # Grid
        self.grid_container = widgets.GridBox(layout=widgets.Layout(grid_template_columns=f"repeat({self.grid_size}, 40px)"))
        
        # Output Area
        self.output_area = widgets.Output()
        
        # Main Layout
        self.main_box = widgets.VBox([
            controls,
            self.palette,
            self.grid_container,
            self.output_area
        ])
        
    def _get_palette_options(self):
        options = [('⬛ Obstacle', 'O'), ('⬜ Free', 'F')]
        for i in range(self.num_agents):
            options.append((f'🟢 S{i+1}', f'S{i}'))
            options.append((f'🏁 G{i+1}', f'G{i}'))
        return options
        
    def _on_agents_changed(self, change):
        self.num_agents = change['new']
        self.palette.options = self._get_palette_options()
        self._initialize_grid() # Re-initialize the grid to show/hide new starts/goals
        
    def _initialize_grid(self):
        self.grid_state.clear()
        self.buttons.clear()
        
        # Default agents top left / bottom right
        starts = {i: (0, i) for i in range(self.num_agents)}
        goals = {i: (self.grid_size - 1, self.grid_size - 1 - i) for i in range(self.num_agents)}
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                self.grid_state[(x, y)] = 'F'
                for i in range(self.num_agents):
                    if (x, y) == starts[i]:
                        self.grid_state[(x, y)] = f'S{i}'
                    elif (x, y) == goals[i]:
                        self.grid_state[(x, y)] = f'G{i}'
                        
        self._render_grid_buttons()
        
    def _render_grid_buttons(self):
        btn_list = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                btn = widgets.Button(layout=widgets.Layout(width='40px', height='40px'))
                btn.x = x
                btn.y = y
                btn.on_click(self._on_grid_clicked)
                self.buttons[(x, y)] = btn
                self._update_button_visuals(x, y)
                btn_list.append(btn)
        self.grid_container.children = btn_list
        
    def _update_button_visuals(self, x, y):
        state = self.grid_state[(x, y)]
        btn = self.buttons[(x, y)]
        btn.style.button_color = self.colors.get(state, self.colors['F'])
        
        if state == 'F':
            btn.description = ""
        elif state == 'O':
            btn.description = ""
        elif state.startswith('S'):
            btn.description = f"S{int(state[1:])+1}"
            btn.style.text_color = "white"
        elif state.startswith('G'):
            btn.description = f"G{int(state[1:])+1}"
            btn.style.text_color = "white"
            
    def _on_grid_clicked(self, btn):
        tool = self.palette.value
        x, y = btn.x, btn.y
        
        # Enforce uniqueness for starts and goals
        if tool.startswith('S') or tool.startswith('G'):
            for (cx, cy), state in self.grid_state.items():
                if state == tool:
                    self.grid_state[(cx, cy)] = 'F'
                    self._update_button_visuals(cx, cy)
                    
        self.grid_state[(x, y)] = tool
        self._update_button_visuals(x, y)
        
    def _validate_map(self):
        starts = []
        goals = []
        for i in range(self.num_agents):
            s = [pos for pos, state in self.grid_state.items() if state == f'S{i}']
            g = [pos for pos, state in self.grid_state.items() if state == f'G{i}']
            if not s or not g:
                return False, f"Agent {i+1} is missing a start or goal."
            # Convert to (y, x) for row, col?
            # Wait, in visualizer it expects (x, y) or (row, col)? 
            # In GridMap, typically coords are (row, col) or (x, y). Let's check map_creator.py.
            # In map_creator.py: tasks are usually y, x or x, y. 
            # Let's use (y, x) which is standard for (row, col).
            starts.append(s[0])
            goals.append(g[0])
        return True, (tuple(starts), tuple(goals))
        
    def _on_run_clicked(self, b):
        is_valid, result = self._validate_map()
        if not is_valid:
            self.status_label.value = f"Error: {result}"
            return
            
        starts, goals = result
        self.status_label.value = "Running OD-RTDP..."
        
        # Construct GridMap
        grid_rows = []
        obstacles = set()
        free_cells = set()
        
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                state = self.grid_state[(x, y)]
                if state == 'O':
                    row.append('@')
                    obstacles.add((x, y))
                else:
                    row.append('.')
                    free_cells.add((x, y))
            grid_rows.append("".join(row))
            
        grid_map = GridMap(
            name="sandbox",
            path=Path("sandbox.map"),
            width=self.grid_size,
            height=self.grid_size,
            grid=tuple(grid_rows),
            obstacles=frozenset(obstacles),
            free_cells=frozenset(free_cells)
        )
        
        # Create Dummy Tasks (required by MapInstance)
        tasks = []
        for i in range(self.num_agents):
            tasks.append(ScenarioEntry(0, "sandbox", self.grid_size, self.grid_size, starts[i], goals[i], 1.0, Path(""), 0))
            
        map_instance = MapInstance(
            grid_map=grid_map,
            scenario_file=Path("sandbox.scen"),
            starts=starts,
            goals=goals,
            tasks=tuple(tasks)
        )
        
        # Run Logic
        mdp = GridMMDP(map_instance, MMDPConfig(slip_to_stay_probability=0.20))
        heuristic = ShortestPathHeuristic(mdp)
        config = RTDPConfig(time_limit_seconds=10.0, seed=42)
        eval_config = EvaluationConfig(episodes=20, seed=101) # Reduced episodes for quick sandbox feedback
        
        try:
            self.status_label.value = "Status: Running Baseline RTDP..."
            baseline_planner = BaselineRTDP(mdp, heuristic, config)
            with ResourceMonitor() as monitor:
                baseline_result = baseline_planner.solve()
            baseline_mem = monitor.snapshot().peak_rss_delta_mb
            baseline_eval = evaluate_policy(mdp, baseline_planner, eval_config)
            
            self.status_label.value = "Status: Running OD RTDP..."
            od_planner = OperatorDecompositionRTDP(mdp, heuristic, config)
            with ResourceMonitor() as monitor:
                od_result = od_planner.solve()
            od_mem = monitor.snapshot().peak_rss_delta_mb
            od_eval = evaluate_policy(mdp, od_planner, eval_config)
            
            self.status_label.value = "Status: Solved!"
            
            with self.output_area:
                self.output_area.clear_output()
                
                # Render the comparison summary
                import ipywidgets as widgets
                
                summary_html = f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #dee2e6;">
                    <h3 style="margin-top: 0; color: #2c3e50;">Performance Comparison</h3>
                    <table style="width: 100%; text-align: left; border-collapse: collapse;">
                        <tr style="border-bottom: 2px solid #bdc3c7;">
                            <th style="padding: 8px;">Algorithm</th>
                            <th style="padding: 8px;">Success Rate</th>
                            <th style="padding: 8px;">Bellman Backups</th>
                            <th style="padding: 8px;">Peak RAM</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #ecf0f1;">
                            <td style="padding: 8px; font-weight: bold; color: #e74c3c;">Baseline (Joint)</td>
                            <td style="padding: 8px;">{baseline_eval.summary.success_rate*100:.1f}%</td>
                            <td style="padding: 8px;">{baseline_result.bellman_backups:,}</td>
                            <td style="padding: 8px;">{baseline_mem:.1f} MB</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold; color: #2ecc71;">Operator Decomposition</td>
                            <td style="padding: 8px;">{od_eval.summary.success_rate*100:.1f}%</td>
                            <td style="padding: 8px;">{od_result.bellman_backups:,}</td>
                            <td style="padding: 8px;">{od_mem:.1f} MB</td>
                        </tr>
                    </table>
                </div>
                """
                display(widgets.HTML(summary_html))
                
                # Visualize the trajectory using the OD planner
                viz = TrajectoryVisualizer(mdp, od_planner, max_steps=100)
                viz.show_with_tree()
        except Exception as e:
            self.status_label.value = f"Error: {str(e)}"
            with self.output_area:
                self.output_area.clear_output()
                print(f"Exception during planning: {e}")

    def show(self):
        display(self.main_box)
