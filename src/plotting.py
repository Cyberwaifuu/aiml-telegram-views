"""Общий стиль графиков: сдержанные оси, палитра серий, русский формат чисел."""

import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from common import RESULTS_DIR, format_ru

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834")
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#a8282a", "#e34948", "#f0efec", "#2a78d6", "#184f95"]
)


def apply_style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
        }
    )


def ru_ticks(axis: Axis, digits: int = 0) -> None:
    axis.set_major_formatter(FuncFormatter(lambda value, _: format_ru(value, digits)))


def save_figure(fig: Figure, name: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / name, bbox_inches="tight")
    plt.close(fig)
