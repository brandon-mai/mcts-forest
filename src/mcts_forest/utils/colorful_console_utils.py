import matplotlib.pyplot as plt
import numpy as np
from typing import Any

# ANSI Color Codes
CEND = "\33[0m"
CBOLD = "\33[1m"
CITALIC = "\33[3m"
CURL = "\33[4m"
CBLINK = "\33[5m"
CBLINK2 = "\33[6m"
CSELECTED = "\33[7m"

CBLACK = "\33[30m"
CRED = "\33[31m"
CGREEN = "\33[32m"
CYELLOW = "\33[33m"
CBLUE = "\33[34m"
CCYAN = "\33[96m"
CMAGENTA = "\033[35m"
CVIOLET = "\33[35m"
CBEIGE = "\33[36m"
CWHITE = "\33[37m"

CBLACKBG = "\33[40m"
CREDBG = "\33[41m"
CGREENBG = "\33[42m"
CYELLOWBG = "\33[43m"
CBLUEBG = "\33[44m"
CVIOLETBG = "\33[45m"
CBEIGEBG = "\33[46m"
CWHITEBG = "\33[47m"

CGREY = "\33[90m"
CRED2 = "\33[91m"
CGREEN2 = "\33[92m"
CYELLOW2 = "\33[93m"
CBLUE2 = "\33[94m"
CCYAN2 = "\033[36m"
CVIOLET2 = "\33[95m"
CBEIGE2 = "\33[96m"
CWHITE2 = "\33[97m"

CGREYBG = "\33[100m"
CREDBG2 = "\33[101m"
CGREENBG2 = "\33[102m"
CYELLOWBG2 = "\33[103m"
CBLUEBG2 = "\33[104m"
CVIOLETBG2 = "\33[105m"
CBEIGEBG2 = "\33[106m"
CWHITEBG2 = "\33[107m"


def rgb_color_sequence(r: int | float, g: int | float, b: int | float,
                       *, format_type: str = 'foreground') -> str:
    """
    Generates a color-codes, that change the color of text in console outputs.
    """
    if format_type == 'foreground':
        f = '\033[38;2;{};{};{}m'.format
    elif format_type == 'background':
        f = '\033[48;2;{};{};{}m'.format
    else:
        raise ValueError(f"format {format_type} is not defined. Use 'foreground' or 'background'.")
    
    rgb = [r, g, b]

    if isinstance(r, int) and isinstance(g, int) and isinstance(b, int):
        if min(rgb) < 0 or max(rgb) > 255:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(r, g, b)
    if isinstance(r, float) and isinstance(g, float) and isinstance(b, float):
        if min(rgb) < 0 or max(rgb) > 1:
            raise ValueError("rgb values must be numbers between 0 and 255 or 0.0 and 1.0")
        return f(*[int(n * 255) for n in [r, g, b]])
    
    return f(0, 0, 0) # Fallback


def wrap_with_color_codes(s: object, /, r: int | float, g: int | float, b: int | float, **kwargs) -> str:
    """
    Stringify an object and wrap it with console color codes.
    """
    return f"{rgb_color_sequence(r, g, b, **kwargs)}{s}{CEND}"


def wrap_evenly_spaced_color(s: Any, n_of_item: int, n_classes: int, c_map="rainbow") -> str:
    """
    Wraps a string with a color scale based on the n_of_item and n_classes.
    Used for coloring actions in the MCTS tree visualization.
    """
    if s is None or n_of_item is None or n_classes is None:
        return str(s)

    try:
        cmap_obj = plt.colormaps.get_cmap(c_map)
    except AttributeError:
        # Fallback for older matplotlib
        cmap_obj = plt.cm.get_cmap(c_map)

    arr = np.linspace(0, 1, n_classes + 1)
    color_vals = cmap_obj(arr[n_of_item])[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"


def wrap_with_color_scale(s: str, value: float, min_val: float, max_val: float, c_map=None) -> str:
    """
    Wraps a string with a color scale based on the value, min_val, and max_val.
    """
    if s is None or min_val is None or max_val is None or min_val >= max_val:
        return str(s)

    if c_map is not None:
        try:
            cmap_obj = plt.colormaps.get_cmap(c_map)
        except AttributeError:
            cmap_obj = plt.cm.get_cmap(c_map)
    else:
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            np.array([255 / 255, 100 / 255, 128 / 255, 1.0]),  # Reddish
            np.array([63 / 255, 197 / 255, 161 / 255, 1.0]),  # Greenish
        ]
        cmap_obj = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Normalize value to [0, 1]
    norm_val = (value - min_val) / (max_val - min_val)
    norm_val = max(0.0, min(1.0, norm_val))
    
    color_vals = cmap_obj(norm_val)[:-1]
    color_asni = rgb_color_sequence(*color_vals, format_type='foreground')

    return f"{color_asni}{s}{CEND}"
