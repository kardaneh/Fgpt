# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

import argparse
import os

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


def read_optimization_history(filename="w_d_rs_optimization_history.txt", logger=None):
    """Read the optimization history file from the Fortran code."""
    try:
        data = np.loadtxt(filename, skiprows=2)
        return {
            "iter": data[:, 0].astype(int),
            "cost": data[:, 1],
            "grad_norm": data[:, 2],
            "lambda_lm": data[:, 3],
            "step_norm": data[:, 4],
            "max_gradient": data[:, 5],
        }
    except Exception as e:
        if logger:
            logger.error(f"Error reading {filename}: {e}")
        else:
            print(f"Error reading {filename}: {e}")
        return None


def read_recovered_parameters(filename="w_d_rs_recovered.txt", logger=None):
    """Read the recovered parameters file from the Fortran code."""
    params = {}
    try:
        with open(filename) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    values = parts[1].strip().split()
                    if len(values) >= 3:
                        params[key] = {
                            "recovered": float(values[0]),
                            "true": float(values[1]),
                            "error": float(values[2]),
                        }
                    elif len(values) >= 1:
                        try:
                            params[key] = {
                                "recovered": float(values[0]),
                                "true": float(values[1]) if len(values) > 1 else None,
                                "error": float(values[2]) if len(values) > 2 else None,
                            }
                        except (ValueError, IndexError):
                            pass
    except Exception as e:
        if logger:
            logger.error(f"Error reading {filename}: {e}")
        else:
            print(f"Error reading {filename}: {e}")
    return params


def read_true_parameters(filename="w_d_rs_true.txt", logger=None):
    """Read the true parameters file from the Fortran code."""
    params = {}
    try:
        with open(filename) as f:
            for line in f:
                if "=" in line:
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            params[key] = float(parts[1].strip())
                        except ValueError:
                            pass
    except Exception as e:
        if logger:
            logger.error(f"Error reading {filename}: {e}")
        else:
            print(f"Error reading {filename}: {e}")
    return params


def read_flux_comparison(filename="w_d_rs_flux_comparison.txt", logger=None):
    """Read the flux comparison file from the Fortran code."""
    try:
        data = np.loadtxt(filename, skiprows=2)
        return {
            "site": data[:, 0].astype(int),
            "day": data[:, 1].astype(int),
            "fup_model": data[:, 2],
            "fup_obs": data[:, 3],
            "residual": data[:, 4],
        }
    except Exception as e:
        if logger:
            logger.error(f"Error reading {filename}: {e}")
        else:
            print(f"Error reading {filename}: {e}")
        return None


def plot_optimization_history(history, logger=None):
    if history is None:
        if logger:
            logger.warning("No history data to plot")
        return None

    fig = Figure(figsize=(14, 10), dpi=300)
    canvas = FigureCanvasAgg(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    linestyles = mpltex.linestyle_generator()
    ax1.semilogy(history["iter"], history["cost"], **next(linestyles))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Cost")
    ax1.set_title("Cost Function Evolution")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    linestyles = mpltex.linestyle_generator()
    ax2.semilogy(history["iter"], history["grad_norm"], **next(linestyles))
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("||gradient||")
    ax2.set_title("Gradient Norm")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    linestyles = mpltex.linestyle_generator()
    ax3.semilogy(history["iter"], history["lambda_lm"], **next(linestyles))
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("$\\lambda$")
    ax3.set_title("LM Damping Parameter")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    linestyles = mpltex.linestyle_generator()
    ax4.semilogy(history["iter"], history["step_norm"], **next(linestyles))
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Step Norm")
    ax4.set_title("Parameter Step Norm")
    ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 1])
    linestyles = mpltex.linestyle_generator()
    ax5.semilogy(history["iter"], history["max_gradient"], **next(linestyles))
    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("Max |gradient|")
    ax5.set_title("Maximum Gradient Component")
    ax5.grid(True, alpha=0.3)

    ax6 = fig.add_subplot(gs[1, 2])
    if len(history["cost"]) > 1:
        cost_rel = np.zeros(len(history["cost"]))
        cost_rel[0] = 1.0
        for i in range(1, len(history["cost"])):
            if history["cost"][i - 1] > 1e-30:
                cost_rel[i] = history["cost"][i] / history["cost"][i - 1]
            else:
                cost_rel[i] = cost_rel[i - 1]
        linestyles = mpltex.linestyle_generator()
        ax6.semilogy(history["iter"][1:], cost_rel[1:], **next(linestyles))
    ax6.set_xlabel("Iteration")
    ax6.set_ylabel("Cost / Cost$_{prev}$")
    ax6.set_title("Relative Cost Reduction")
    ax6.grid(True, alpha=0.3)
    canvas.print_figure("optimization_history_b.png", dpi=300, bbox_inches="tight")
    if logger:
        logger.info("  - optimization_history_b.png")
    return fig


def plot_parameters_summary(recovered_params, true_params, logger=None):
    if not recovered_params:
        if logger:
            logger.warning("No recovered parameters to plot")
        return None

    fig = Figure(figsize=(12, 8), dpi=300)
    canvas = FigureCanvasAgg(fig)

    param_names = []
    recovered_vals = []
    true_vals = []
    errors = []

    if "w_d" in recovered_params:
        param_names.append("$w$")
        recovered_vals.append(recovered_params["w_d"]["recovered"])
        true_vals.append(
            recovered_params["w_d"]["true"]
            if recovered_params["w_d"]["true"] is not None
            else 0
        )
        errors.append(
            recovered_params["w_d"]["error"]
            if recovered_params["w_d"]["error"] is not None
            else 0
        )

    if "d" in recovered_params:
        param_names.append("$d$")
        recovered_vals.append(recovered_params["d"]["recovered"])
        true_vals.append(
            recovered_params["d"]["true"]
            if recovered_params["d"]["true"] is not None
            else 0
        )
        errors.append(
            recovered_params["d"]["error"]
            if recovered_params["d"]["error"] is not None
            else 0
        )

    rs_keys = sorted([k for k in recovered_params.keys() if k.startswith("rs(")])
    for key in rs_keys:
        site_idx = key.split("(")[1].split(")")[0]
        param_names.append(f"$rs_{{{site_idx}}}$")
        recovered_vals.append(recovered_params[key]["recovered"])
        true_vals.append(recovered_params[key]["true"])
        errors.append(recovered_params[key]["error"])

    ax = fig.add_subplot(111)
    x = np.arange(len(param_names))
    width = 0.35

    ax.bar(x - width / 2, true_vals, width, label="True", alpha=0.7, color="green")
    ax.bar(
        x + width / 2, recovered_vals, width, label="Recovered", alpha=0.7, color="blue"
    )
    ax.errorbar(
        x + width / 2,
        recovered_vals,
        yerr=errors,
        fmt="none",
        ecolor="red",
        capsize=3,
        capthick=1,
    )

    ax.set_xlabel("Parameter")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    canvas.print_figure("parameters_summary_b.png", dpi=300, bbox_inches="tight")
    if logger:
        logger.info("  - parameters_summary_b.png")
    return fig


def plot_flux_comparison(fluxes, n_sites=8, logger=None):
    if fluxes is None:
        if logger:
            logger.warning("No flux data to plot")
        return None

    n_cols = 6
    n_rows = (n_sites + n_cols - 1) // n_cols
    fig = Figure(figsize=(n_cols * 5, n_rows * 4), dpi=300)
    canvas = FigureCanvasAgg(fig)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)

    unique_sites = np.unique(fluxes["site"])

    for idx, site in enumerate(unique_sites):
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        mask = fluxes["site"] == site
        day = fluxes["day"][mask]
        fup_model = fluxes["fup_model"][mask]
        fup_obs = fluxes["fup_obs"][mask]
        residual = fluxes["residual"][mask]

        linestyles = mpltex.linestyle_generator(markers=[])
        ax.plot(day, fup_obs, label="Observed", **next(linestyles))
        ax.plot(day, fup_model, label="Recovered", **next(linestyles))
        ax.set_xlabel("Day")
        ax.set_ylabel("Upward Flux")
        ax.set_title(f"Site {site} (RMSE={np.sqrt(np.mean(residual**2)):.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    canvas.print_figure("flux_comparison_b.png", dpi=300, bbox_inches="tight")
    if logger:
        logger.info("  - flux_comparison_b.png")
    return fig


def print_summary(recovered_params, true_params, logger):
    logger.show_header("PARAMETER RECOVERY SUMMARY")

    total_error = 0.0
    total_rel_error = 0.0
    count = 0

    if "w_d" in recovered_params:
        w = recovered_params["w_d"]
        w_true = true_params.get("True_w", w["true"] if w["true"] is not None else 0)
        w_rec = w["recovered"]
        w_err = abs(w_rec - w_true)
        w_rel = w_err / max(abs(w_true), 1e-12) * 100
        logger.info("")
        logger.info("w (Single Scattering Albedo):")
        logger.info(f"  True:      {w_true:.8f}")
        logger.info(f"  Recovered: {w_rec:.8f}")
        logger.info(f"  Error:     {w_err:.6e} ({w_rel:.4f}%)")
        total_error += w_err
        total_rel_error += w_rel
        count += 1

    if "d" in recovered_params:
        d = recovered_params["d"]
        d_true = true_params.get("True_d", d["true"] if d["true"] is not None else 0)
        d_rec = d["recovered"]
        d_err = abs(d_rec - d_true)
        d_rel = d_err / max(abs(d_true), 1e-12) * 100
        logger.info("")
        logger.info("d (Shape Parameter):")
        logger.info(f"  True:      {d_true:.8f}")
        logger.info(f"  Recovered: {d_rec:.8f}")
        logger.info(f"  Error:     {d_err:.6e} ({d_rel:.4f}%)")
        total_error += d_err
        total_rel_error += d_rel
        count += 1

    logger.info("")
    logger.info("rs (Surface Reflectance per Site):")
    rs_keys = sorted([k for k in recovered_params.keys() if k.startswith("rs(")])
    for key in rs_keys:
        rs = recovered_params[key]
        site_idx = key.split("(")[1].split(")")[0]
        rs_true = true_params.get(f"True_rs({site_idx})", rs["true"])
        rs_rec = rs["recovered"]
        rs_err = abs(rs_rec - rs_true)
        rs_rel = rs_err / max(abs(rs_true), 1e-12) * 100
        logger.info(f"  rs({site_idx}):")
        logger.info(f"    True:      {rs_true:.8f}")
        logger.info(f"    Recovered: {rs_rec:.8f}")
        logger.info(f"    Error:     {rs_err:.6e} ({rs_rel:.4f}%)")
        total_error += rs_err
        total_rel_error += rs_rel
        count += 1

    if count > 0:
        logger.info("")
        logger.info(f"Average absolute error:   {total_error / count:.6e}")
        logger.info(f"Average relative error:   {total_rel_error / count:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Plot optimization results for w,d,rs calibration"
    )
    parser.add_argument("--n_sites", type=int, default=8, help="Number of sites")
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Output directory for plots"
    )
    args = parser.parse_args()

    logger = Logger()
    logger.show_header("W, D, RS CALIBRATION - PLOTTING TOOL")

    if args.output_dir != ".":
        os.chdir(args.output_dir)
        logger.info(f"Output directory: {args.output_dir}")

    logger.info("Reading data files...")
    history = read_optimization_history("w_d_rs_optimization_history.txt", logger)
    true_params = read_true_parameters("w_d_rs_true.txt", logger)
    recovered_params = read_recovered_parameters("w_d_rs_recovered.txt", logger)
    fluxes = read_flux_comparison("w_d_rs_flux_comparison.txt", logger)

    if recovered_params and true_params:
        print_summary(recovered_params, true_params, logger)
    else:
        logger.warning("Missing parameter data for summary")

    logger.info("")
    logger.info("Generating plots...")

    plot_optimization_history(history, logger)

    plot_parameters_summary(recovered_params, true_params, logger)

    plot_flux_comparison(fluxes, args.n_sites, logger)

    logger.info("")
    logger.show_header("PLOTS COMPLETE")
    logger.info("Generated files:")
    logger.info("  - optimization_history_b.png")
    logger.info("  - parameters_summary_b.png")
    logger.info("  - flux_comparison_b.png")


if __name__ == "__main__":
    main()
