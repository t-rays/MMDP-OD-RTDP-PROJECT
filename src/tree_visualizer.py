import graphviz
from IPython.display import SVG, HTML
import ipywidgets as widgets

class BranchingTreeVisualizer:
    @staticmethod
    def generate_trees_svg(n_agents: int) -> tuple[str, str]:
        # 1. Joint Action Tree
        no_od = graphviz.Digraph(format='svg')
        title_no = f'WITHOUT OD (Joint Space)\nBranching = 5^{n_agents} = {5**n_agents:,}\nAll joint actions evaluated at once.'
        no_od.attr(rankdir='LR', label=title_no, fontname='Helvetica-Bold', fontcolor='#d63031')
        no_od.node('S', 'State', shape='box', style='filled', fillcolor='#fab1a0')
        
        # Draw a subset of joint actions
        no_od.node('A1', f'Joint Action 1', shape='box', style='filled', fillcolor='#ff7675')
        no_od.node('A2', f'...', shape='none')
        no_od.node('A3', f'Joint Action {5**n_agents:,}', shape='box', style='filled', fillcolor='#ff7675')
        no_od.edge('S', 'A1')
        no_od.edge('S', 'A2', style='dotted')
        no_od.edge('S', 'A3')
        
        # 2. OD Tree
        od = graphviz.Digraph(format='svg')
        title_od = f'WITH OD (Sequential)\nBranching = 5 × {n_agents} = {5 * n_agents}\nAgents decide sequentially.'
        od.attr(rankdir='LR', label=title_od, fontname='Helvetica-Bold', fontcolor='#0984e3')
        
        prev = 'S'
        od.node(prev, 'State', shape='box', style='filled', fillcolor='#74b9ff')
        
        for i in range(1, n_agents + 1):
            curr = f'A{i}'
            od.node(curr, f'Agent {i}\n(5 moves)', shape='box', style='filled', fillcolor='#0984e3', fontcolor='white')
            od.edge(prev, curr)
            prev = curr
            
        return no_od.pipe().decode('utf-8'), od.pipe().decode('utf-8')
