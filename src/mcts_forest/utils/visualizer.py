import os
import numpy as np
from typing import List, Any, Optional, Generator, Tuple, Dict


class SurrogateNode:
    """Virtual node that wraps Numba/NumPy data for visualization."""
    def __init__(self, node_id: int, visit_count: np.ndarray, q_hat: np.ndarray, child_nodes: dict, action: Any = None, a_visits: np.ndarray = None):
        self.node_id = node_id
        self.v_count = visit_count
        self._q_hat = q_hat
        self._child_nodes = child_nodes
        self.action = action
        self.a_visits = a_visits # Optional action visits array

    @property
    def children(self) -> Dict[Any, 'SurrogateNode']:
        res = {}
        if isinstance(self._child_nodes, np.ndarray):
            # 2D Array case: child_nodes[node_id, action] = next_node_id
            for action in range(self._child_nodes.shape[1]):
                child_id = int(self._child_nodes[self.node_id, action])
                if child_id != -1:
                    child_node = SurrogateNode(child_id, self.v_count, self._q_hat, self._child_nodes, action=action, a_visits=self.a_visits)
                    res[f"{action}_{child_id}"] = child_node
        elif isinstance(self._child_nodes, dict):
            # Dictionary case: key is (parent, action) or (parent, action, next_state)
            for key, child_id in self._child_nodes.items():
                if key[0] == self.node_id:
                    action = key[1]
                    child_node = SurrogateNode(child_id, self.v_count, self._q_hat, self._child_nodes, action=action, a_visits=self.a_visits)
                    res[f"{action}_{child_id}"] = child_node
        return res

    def __str__(self, colored: bool = True, action_space_n: int = 100, c: float = 1.41) -> str:
        from mcts_forest.utils import colorful_console_utils as ccu
        
        n = self.visit_count
        label = f"Node {self.node_id}"
        n_sa = 0
        
        # Access parent's action_visits if possible
        # For simplicity, we'll just show the node's visit count and 
        # its own best Q.
        
        q_vals = self._q_hat[self.node_id]
        best_a = int(np.argmax(q_vals))
        max_q = float(q_vals[best_a])
        
        # If we have a_visits, we can show visits to each action from this node
        a_info = ""
        if self.a_visits is not None:
             v_actions = self.a_visits[self.node_id]
             n_sa = int(v_actions[self.action]) if self.action is not None else 0
             a_info = f" [n_sa={n_sa}]" if self.action is not None else f" [sum_n_sa={int(np.sum(v_actions))}]"

        label_full = f"Action {self.action}" if self.action is not None else f"Node {self.node_id}"
        res = f"{label_full}{a_info} [state_n={n}] Max Q: {max_q:.4f} (a={best_a})"
        
        if self.node_id == 0:
            res += " | Q:" + str([f"{q:.2f}" for q in q_vals[:action_space_n]])
        
        if colored:
            return ccu.wrap_with_color_scale(res, max_q, -20, 20)
        return res

def _generate_mcts_tree(
    node: Any,
    prefix: str = "",
    depth: int = 2,
    colored: bool = True,
    action_space_n: int = 100,
    c: float = 1.41,
    exclude_unvisited: bool = True
) -> Generator[str, None, None]:
    """
    Recursively generates a tree representation of the MCTS tree.
    Matches gymcts_original generator logic.
    """
    from mcts_forest.utils import colorful_console_utils as ccu

    # prefix components (like gymcts_original)
    space = '    '
    branch = '│   '
    tee = '├── '
    last = '└── '

    # Filter and sort children
    children = [n for n in node.children.values() if not exclude_unvisited or n.visit_count > 0]
    children.sort(key=lambda n: n.visit_count, reverse=True)
    
    # Determine pointers
    pointers = [tee] * (len(children) - 1) + [last] if children else []
    
    for pointer, child in zip(pointers, children):
        n_item = child.action if isinstance(child.action, int) else 0
        n_classes = action_space_n
        
        p_str = pointer
        if colored:
            p_str = ccu.wrap_evenly_spaced_color(s=pointer, n_of_item=n_item, n_classes=n_classes)
        
        yield prefix + p_str + child.__str__(colored=colored, action_space_n=n_classes, c=c)
        
        if depth > 0 and child.children:
            # Extension logic
            extension = branch if pointer == tee else space
            if colored:
                extension = ccu.wrap_evenly_spaced_color(s=extension, n_of_item=n_item, n_classes=n_classes)
            
            yield from _generate_mcts_tree(
                child, prefix + extension, depth - 1, colored, action_space_n, c, exclude_unvisited
            )

def print_tree(root: Any, path: Optional[str] = None, action_space_n: int = 100, max_depth: int = 2, c: float = 1.41, **kwargs) -> str:
    """
    Prints the MCTS tree. Returns the monochrome version.
    """
    silent = kwargs.get('silent', False)

    # 1. Colored version for terminal
    colored_root = root.__str__(colored=True, action_space_n=action_space_n, c=c)
    if not silent:
        print("\n--- MCTS TREE ---")
        print(colored_root)
        for line in _generate_mcts_tree(root, depth=max_depth, colored=True, action_space_n=action_space_n, c=c):
            print(line)
        print("-----------------\n")

    # 2. Monochrome version for string/file
    plain_lines = [root.__str__(colored=False, c=c)]
    plain_lines.extend(_generate_mcts_tree(root, depth=max_depth, colored=False, action_space_n=action_space_n, c=c))
    monochrome_str = "\n".join(plain_lines)

    if path:
        with open(path, "a", encoding="utf-8") as f: # Append mode
            f.write(monochrome_str + "\n")
    
    return monochrome_str
