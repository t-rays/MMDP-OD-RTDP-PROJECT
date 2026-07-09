import graphviz
from typing import Optional, Tuple, Dict, Any

ACTIONS = ['N', 'S', 'E', 'W', 'stay']
COLORS = {
    'chosen': '#0984e3',
    'joint_best': '#e17055',
    'rejected': '#dfe6e9',
    'start_no': '#d63031',
    'start_with': '#0984e3',
    'text_light': 'white',
    'unselected': '#b2bec3',
    'agent_2': '#e17055',
    'agent_3': '#00b894'
}
GAMMA = 0.99

def format_graphviz_math(text: str) -> str:
    return text.replace('{', '').replace('}', '')

class TreeVisualizer:
    def __init__(self, num_agents: int, slip: float):
        self.num_agents = num_agents
        self.slip = slip
        self.no_od = graphviz.Digraph(format='svg')
        self.with_od = graphviz.Digraph(format='svg')
        self.current_root_no = 'START_NO'
        self.current_root_with = 'START_WITH'
        self._initialize_graphs()

    def _initialize_graphs(self) -> None:
        n = self.num_agents
        
        # Joint MMDP Tree
        title_no = (f'WITHOUT OD — Joint MMDP:  A = A^1 × ... × A^{n},  |A| = 5^{n} = {5**n:,}\n'
                    f'Tree shows joint actions evaluated each step with '
                    f'JointQ = Σ_k Q^k − conflict penalty  (slip = {self.slip:.0%}, γ = {GAMMA})')
        self.no_od.attr(rankdir='LR', label=format_graphviz_math(title_no),
                        fontname='Helvetica-Bold', fontcolor='#d63031')
        self.no_od.node('START_NO', format_graphviz_math('s_{0} ∈ S'), shape='box',
                        style='filled', fillcolor=COLORS['start_no'], fontcolor=COLORS['text_light'])

        # OD Tree
        title_with = (f'WITH OD — Sequential Decomposition:  a_{{t}}^k = argmax_a Q^k(s_{{t}}, a)  '
                      f's.t. no vertex/swap conflict with a_{{t}}^{{1:k-1}}\n'
                      f'Q^k from value iteration:  Q^k(s,a) = Σ_{{s\'}} T(s\'|s,a)[−1 + γ V^k(s\')]  '
                      f'(genuine conditioning — later agents see earlier reservations)')
        self.with_od.attr(rankdir='LR', label=format_graphviz_math(title_with),
                          fontname='Helvetica-Bold', fontcolor='#0984e3')
        self.with_od.node('START_WITH', format_graphviz_math('s_{0} ∈ S'), shape='box',
                          style='filled', fillcolor=COLORS['start_with'], fontcolor=COLORS['text_light'])

    def add_step(self, step: int, decision: Dict[str, Any], info: Dict[str, Any]) -> Tuple[str, str]:
        acts = decision['od_actions']
        a1 = acts[0] if len(acts) > 0 else 4
        a2 = acts[1] if len(acts) > 1 else 4
        a3 = acts[2] if len(acts) > 2 else 4
        self._update_no_od(step, a1, a2, a3, decision, info)
        self._update_with_od(step, a1, a2, a3, decision, info)
        try:
            return (self.no_od.pipe(format='svg').decode('utf-8'),
                    self.with_od.pipe(format='svg').decode('utf-8'))
        except Exception as e:
            banner = (f"<div style='padding:20px; color:#d63031; font-weight:bold; "
                      f"border:2px dashed #d63031; border-radius:8px;'>"
                      f"⚠ Tree rendering failed: {type(e).__name__}: {e}<br>"
                      f"Most likely the Graphviz system binary ('dot') is not installed "
                      f"or not on PATH.</div>")
            return banner, banner

    def _update_no_od(self, step: int, a1: int, a2: int, a3: int,
                      decision: Dict[str, Any], info: Dict[str, Any]) -> None:
        chosen_idx = decision['chosen_idx']
        best_idx = decision['joint_best_idx']
        scores = decision['joint_scores']
        next_root = f'L_{step}_{chosen_idx}'
        
        n = self.num_agents
        
        # Generalize over agents
        for i in range(5):
            for j in range(5):
                # If only 2 agents, mock k as 0 and only run once
                for k in range(5 if n > 2 else 1):
                    if n == 2:
                        loop_idx = (i * 5) + j
                    else:
                        loop_idx = (i * 25) + (j * 5) + k
                        
                    is_chosen = (loop_idx == chosen_idx)
                    is_best = (loop_idx == best_idx) and (best_idx != chosen_idx)
                    node_name = f'L_{step}_{loop_idx}'
                    
                    if n == 2:
                        tip = f"({ACTIONS[i]},{ACTIONS[j]})  JointQ={scores[loop_idx]:.2f}"
                        a_str = f"({ACTIONS[a1]}, {ACTIONS[a2]})"
                        plcholder = f"{ACTIONS[i][0]}{ACTIONS[j][0]}"
                    else:
                        tip = f"({ACTIONS[i]},{ACTIONS[j]},{ACTIONS[k]})  JointQ={scores[loop_idx]:.2f}"
                        a_str = f"({ACTIONS[a1]}, {ACTIONS[a2]}, {ACTIONS[a3]})"
                        plcholder = f"{ACTIONS[i][0]}{ACTIONS[j][0]}{ACTIONS[k][0]}"

                    if is_chosen:
                        lbl = (f"s_{{{step}}} ∈ S\n"
                               f"a_{{{step-1}}} = {a_str}\n"
                               f"JointQ = {scores[chosen_idx]:.2f}   (idx {chosen_idx} / {5**n} evaluated)")
                        if best_idx == chosen_idx:
                            lbl += "\n= true joint argmax ✓"
                        else:
                            lbl += f"\njoint argmax differs (idx {best_idx}, orange)"
                        slips = [f"A{x+1}:{ACTIONS[acts_i]}→{ACTIONS[info['executed'][x]]}"
                                 for x, acts_i in enumerate(decision['od_actions'])
                                 if x < len(info['slipped']) and info['slipped'][x]]
                        if slips:
                            lbl += "\nslipped: " + ", ".join(slips)
                        self.no_od.node(node_name, format_graphviz_math(lbl), shape='box',
                                        style='filled', fillcolor=COLORS['chosen'],
                                        fontsize='9', tooltip=tip)
                    elif is_best:
                        lbl = (f"joint argmax\n{plcholder}\n"
                               f"JointQ = {scores[loop_idx]:.2f}")
                        self.no_od.node(node_name, format_graphviz_math(lbl), shape='box',
                                        style='filled', fillcolor=COLORS['joint_best'],
                                        fontcolor=COLORS['text_light'], fontsize='8', tooltip=tip)
                    else:
                        self.no_od.node(node_name, plcholder, shape='circle', style='filled',
                                        fillcolor=COLORS['rejected'], fontsize='5',
                                        width='0.25', tooltip=tip)

                    color = COLORS['chosen'] if is_chosen else (
                        COLORS['joint_best'] if is_best else COLORS['rejected'])
                    width = '3.5' if (is_chosen or is_best) else '1.0'
                    self.no_od.edge(self.current_root_no, node_name, color=color,
                                    penwidth=width, arrowhead='none')

        self.current_root_no = next_root

    def _update_with_od(self, step: int, a1: int, a2: int, a3: int,
                        decision: Dict[str, Any], info: Dict[str, Any]) -> None:
        od_q = decision['od_q']
        reserved = decision['reserved_targets']
        n = self.num_agents

        # Agent 1 micro-decision
        for i in range(5):
            is_a1 = (i == a1)
            c1 = COLORS['chosen'] if is_a1 else COLORS['unselected']
            w1 = '3.5' if is_a1 else '1.0'
            node_a1 = f'OD_{step}_a1_{i}'
            tip1 = f"Q^1({ACTIONS[i]}) = {od_q[0][i]:.2f}" if len(od_q) > 0 else ACTIONS[i]

            if is_a1:
                lbl = (f"a^1_{{{step-1}}} = {ACTIONS[a1]} = argmax_a Q^1(s_{{{step-1}}}, a)\n"
                       f"Q^1 = {od_q[0][a1]:.2f}   reserves cell {reserved[0]}")
                self.with_od.node(node_a1, format_graphviz_math(lbl), shape='box',
                                  style='filled', fillcolor=c1, fontsize='8', tooltip=tip1)
            else:
                self.with_od.node(node_a1, ACTIONS[i][0], shape='circle', style='filled',
                                  fillcolor=c1, fontsize='6', width='0.2', tooltip=tip1)
            self.with_od.edge(self.current_root_with, node_a1, color=c1,
                              penwidth=w1, arrowhead='none')

            if not is_a1:
                continue

            # Agent 2 micro-decision
            for j in range(5):
                is_a2 = (j == a2)
                c2 = COLORS['chosen'] if is_a2 else COLORS['unselected']
                w2 = '3.5' if is_a2 else '1.0'
                node_a2 = f'OD_{step}_a2_{j}'
                tip2 = f"Q^2({ACTIONS[j]}) = {od_q[1][j]:.2f}" if len(od_q) > 1 else ACTIONS[j]

                if is_a2:
                    lbl = (f"a^2_{{{step-1}}} = {ACTIONS[a2]} ~ π^2(·|s_{{{step-1}}}, a^1)\n"
                           f"Q^2 = {od_q[1][a2]:.2f}, avoids reserved {reserved[0]}\n"
                           f"reserves cell {reserved[1]}")
                    if n == 2:
                        lbl += f"\ns_{{{step}}} = T(s_{{{step-1}}}, a_{{{step-1}}})  (slip = {self.slip:.0%})"
                    
                    self.with_od.node(node_a2, format_graphviz_math(lbl), shape='diamond',
                                      style='filled', fillcolor=COLORS['agent_2'],
                                      fontcolor=COLORS['text_light'], fontsize='8', tooltip=tip2)
                else:
                    self.with_od.node(node_a2, ACTIONS[j][0], shape='circle', style='filled',
                                      fillcolor=c2, fontsize='6', width='0.2', tooltip=tip2)
                self.with_od.edge(node_a1, node_a2, color=c2, penwidth=w2, arrowhead='none')

                if not is_a2 or n == 2:
                    continue

                # Agent 3 micro-decision (if n == 3)
                for k in range(5):
                    is_a3 = (k == a3)
                    c3 = COLORS['chosen'] if is_a3 else COLORS['agent_3']
                    w3 = '3.5' if is_a3 else '1.0'
                    node_a3 = f'OD_{step}_a3_{k}'
                    tip3 = f"Q^3({ACTIONS[k]}) = {od_q[2][k]:.2f}" if len(od_q) > 2 else ACTIONS[k]

                    if is_a3:
                        lbl = (f"a^3_{{{step-1}}} = {ACTIONS[a3]} ~ π^3(·|s_{{{step-1}}}, a^1, a^2)\n"
                               f"Q^3 = {od_q[2][a3]:.2f}, avoids {{{reserved[0]}, {reserved[1]}}}\n"
                               f"s_{{{step}}} = T(s_{{{step-1}}}, a_{{{step-1}}})  (slip = {self.slip:.0%})")
                        self.with_od.node(node_a3, format_graphviz_math(lbl), shape='box',
                                          style='filled', fillcolor=c3, fontsize='8', tooltip=tip3)
                    else:
                        self.with_od.node(node_a3, ACTIONS[k][0], shape='circle', style='filled',
                                          fillcolor=c3, fontsize='6', width='0.2', tooltip=tip3)
                    self.with_od.edge(node_a2, node_a3, color=c3, penwidth=w3, arrowhead='none')

        if n == 3:
            self.current_root_with = f'OD_{step}_a3_{a3}'
        elif n == 2:
            self.current_root_with = f'OD_{step}_a2_{a2}'
