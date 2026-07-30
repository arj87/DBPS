"""
utils.py
--------
Small shared helpers used across modules.
"""

import matplotlib.pyplot as plt

DEFAULT_STYLE = {
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "font.size": 11,
}


def apply_style():
    plt.rcParams.update(DEFAULT_STYLE)


def save_fig(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[utils] Saved figure -> {path}")
