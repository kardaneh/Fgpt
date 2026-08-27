# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import mpltex
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from fgpt.core.common.logger import Logger

params = {
    "font.family": "DejaVu Sans",
    "lines.linewidth": 2,
    "lines.dashed_pattern": [4, 2],
    "lines.dashdot_pattern": [6, 3, 2, 3],
    "lines.dotted_pattern": [2, 3],
    "mathtext.rm": "arial",
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "legend.fontsize": 15,
    "legend.loc": "best",
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
}
mpl.rcParams.update(params)


def read_optimization_history(filename="optimization_history.txt"):
    data = np.loadtxt(filename, skiprows=2)
    return {
        "iter": data[:, 0].astype(int),
        "cost": data[:, 1],
        "w": data[:, 2],
        "d": data[:, 3],
        "gJ_w": data[:, 4],
        "gJ_d": data[:, 5],
        "lambda": data[:, 6],
        "rho": data[:, 7],
    }


def read_true_parameters(filename="true_parameters.txt"):
    data = {}
    with open(filename) as f:
        for line in f:
            if "=" in line:
                parts = line.strip().split("=")
                if len(parts) == 2:
                    key = parts[0].strip()
                    try:
                        data[key] = float(parts[1].strip())
                    except ValueError:
                        pass
    return data


def plot_optimization_history(history, true_w=0.2, true_d=0.8):
    fig = Figure(figsize=(12, 14), dpi=300)
    canvas = FigureCanvasAgg(fig)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    linestyles = mpltex.linestyle_generator()
    ax1.semilogy(history["iter"], history["cost"], **next(linestyles))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost Function Evolution")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    linestyles = mpltex.linestyle_generator()
    ax2.plot(history["iter"], history["w"], label="w (recovered)", **next(linestyles))
    ax2.plot(history["iter"], history["d"], label="d (recovered)", **next(linestyles))
    ax2.axhline(y=true_w, label=f"True w = {true_w:.4f}", **next(linestyles))
    ax2.axhline(y=true_d, label=f"True d = {true_d:.4f}", **next(linestyles))
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Parameter value")
    ax2.set_title("Parameter Evolution")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()
    ax3.semilogy(
        history["iter"], np.abs(history["gJ_w"]), label="|gJ_w|", **next(linestyles)
    )
    ax3.semilogy(
        history["iter"], np.abs(history["gJ_d"]), label="|gJ_d|", **next(linestyles)
    )
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Gradient magnitude")
    ax3.set_title("Gradient Evolution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    linestyles = mpltex.linestyle_generator()
    ax4.semilogy(history["iter"], history["lambda"], **next(linestyles))
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("$\\lambda$")
    ax4.set_title("Levenberg-Marquardt Damping Parameter")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    linestyles = mpltex.linestyle_generator()
    ax5.semilogy(
        history["iter"],
        np.abs(history["w"] - true_w),
        label="w error",
        **next(linestyles),
    )
    ax5.semilogy(
        history["iter"],
        np.abs(history["d"] - true_d),
        label="d error",
        **next(linestyles),
    )
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Absolute error")
    ax5.set_title("Parameter Error Evolution")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    linestyles = mpltex.linestyle_generator()
    mask = history["rho"] >= 0
    ax6.plot(history["iter"][mask], history["rho"][mask], **next(linestyles))
    ax6.axhline(y=1, **next(linestyles))
    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("$\\rho$")
    ax6.set_title("Gain Ratio ($\\rho$)")
    ax6.grid(True, alpha=0.3)
    canvas.print_figure("optimization_history.png", dpi=300, bbox_inches="tight")
    return fig


def log_summary(logger, history, true_w=0.2, true_d=0.8):
    final_cost = history["cost"][-1]
    final_w = history["w"][-1]
    final_d = history["d"][-1]

    logger.info("=" * 60)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {len(history['iter'])}")
    logger.info(f"Final cost: {final_cost:.6e}")
    logger.info("Parameter Recovery:")
    logger.info("")
    logger.info(f"  w_true = {true_w:.6f}")
    logger.info(f"  w_recovered = {final_w:.6f}")
    logger.info(
        f"  w_error = {abs(final_w - true_w):.6e} ({abs(final_w - true_w) / max(true_w, 1e-12) * 100:.4f}%)"
    )
    logger.info("")
    logger.info(f"  d_true = {true_d:.6f}")
    logger.info(f"  d_recovered = {final_d:.6f}")
    logger.info(
        f"  d_error = {abs(final_d - true_d):.6e} ({abs(final_d - true_d) / max(true_d, 1e-12) * 100:.4f}%)"
    )
    logger.info("")
    logger.info(
        f"  Final gradient norm: {np.sqrt(history['gJ_w'][-1] ** 2 + history['gJ_d'][-1] ** 2):.6e}"
    )
    logger.info(f"  Final lambda: {history['lambda'][-1]:.6e}")
    logger.info("=" * 60)


def main():
    try:
        logger = Logger()
        logger.show_header("Plot initialized")

        history = read_optimization_history("optimization_history.txt")
        true_params = read_true_parameters("true_parameters.txt")
        true_w = true_params.get("True_w", 0.25)
        true_d = true_params.get("True_d", 0.6)

        logger.info(f"True parameters from file: w={true_w:.6f}, d={true_d:.6f}")
        logger.info("Generating optimization history plots...")

        plot_optimization_history(history, true_w, true_d)
        log_summary(logger, history, true_w, true_d)

        logger.info("Plots saved as:")
        logger.info("  - optimization_history.png")

    except FileNotFoundError as e:
        logger.info(
            "Error: Could not find data file. Make sure you run the Fortran code first."
        )
        logger.error(f"Missing file: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    main()
