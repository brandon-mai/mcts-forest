from typing import Dict, Any, List
import ast

def parse_dict(arg: str) -> Dict[str, Any]:
    """
    Safely parse a string representation of a dictionary using ast.literal_eval.
    """
    if not arg or arg == "{}":
        return {}
    try:
        res = ast.literal_eval(arg)
        if isinstance(res, dict):
            return res
        return {}
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse dictionary from string: {arg}. Error: {e}")
        return {}

def generate_experiment_name(env: Any, solver: Any, seeds: int = 1):
    """
    Standardized experiment name: {env_name}_{solver_with_params}_s{seeds}
    """
    env_name = env.get_name() if hasattr(env, 'get_name') else str(env)
    solver_name = solver.get_name() if hasattr(solver, 'get_name') else str(solver)
    
    return f"{env_name}_{solver_name}_s{seeds}"

def clear_directory(path: str):
    """
    Safely delete a directory and its contents if it exists.
    """
    import shutil
    import os
    if os.path.exists(path):
        print(f"Clearing directory: {path}")
        shutil.rmtree(path)
