import graphviz
from typing import Optional, Tuple

class BranchingTreeVisualizer:
    @staticmethod
    def generate_trees_svg(n_agents: int, step: int = 0, action: Optional[Tuple[str, ...]] = None) -> tuple[str, str]:
        # 1. Joint Action Tree
        no_od = graphviz.Digraph(format='svg')
        title_no = f'WITHOUT OD (Joint Space)\nBranching = 5^{n_agents} = {5**n_agents:,}\nStep: {step}'
        no_od.attr(rankdir='LR', label=title_no, fontname='Helvetica-Bold', fontcolor='#d63031')
        no_od.node('S', f'State at t={step}', shape='box', style='filled', fillcolor='#fab1a0')
        
        # Draw a subset of joint actions
        if action:
            act_str = ",".join(a[0].upper() for a in action)
            no_od.node('A_chosen', f'Chosen Joint Action\n({act_str})', shape='box', style='filled', fillcolor='#d63031', fontcolor='white')
            no_od.edge('S', 'A_chosen', color='#d63031', penwidth='3.0')
            
        no_od.node('A_other', f'Other {5**n_agents - 1:,} Joint Actions...', shape='none')
        no_od.edge('S', 'A_other', style='dotted', color='#b2bec3')
        
        # 2. OD Tree
        od = graphviz.Digraph(format='svg')
        title_od = f'WITH OD (Sequential)\nBranching = 5 × {n_agents} = {5 * n_agents}\nStep: {step}'
        od.attr(rankdir='LR', label=title_od, fontname='Helvetica-Bold', fontcolor='#0984e3')
        
        prev = 'S'
        od.node(prev, f'State at t={step}', shape='box', style='filled', fillcolor='#74b9ff')
        
        for i in range(n_agents):
            curr = f'A{i+1}'
            if action and i < len(action):
                act = action[i]
                od.node(curr, f'Agent {i+1} locked:\n{act}', shape='box', style='filled', fillcolor='#0984e3', fontcolor='white')
                od.edge(prev, curr, color='#0984e3', penwidth='3.0')
                
                # Draw the unchosen 4 branches lightly
                other = f'Other{i+1}'
                od.node(other, '4 unused actions', shape='none', fontcolor='#b2bec3')
                od.edge(prev, other, style='dotted', color='#b2bec3')
            else:
                od.node(curr, f'Agent {i+1}\n(5 moves)', shape='box', style='filled', fillcolor='#74b9ff')
                od.edge(prev, curr)
            prev = curr
            
        return no_od.pipe().decode('utf-8'), od.pipe().decode('utf-8')
