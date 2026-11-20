#!/usr/bin/env python3

# Copyright 2026 CNRS
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Multi-site calibration of (w, d) for the two-stream radiative transfer model,
using the JAX forward model and a Levenberg-Marquardt optimizer.

Python/JAX port of calibrate.F90 (twin experiment):
  - synthetic references are generated with random (w_true, d_true)
  - the Jacobian comes from jax.jacfwd instead of Tapenade tangent code
  - outputs match the Fortran ones (true_parameters.txt,
    optimization_history.txt, fluxes_comparison.txt) so plots.py still works

Usage:
    python calibrate.py --n_sites 8 [--seed 0]
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from global_module_multilevel_matrix_d import Global_module_multilevel_matrix_eqx

jax.config.update("jax_enable_x64", True)

# Settings (same as the Fortran program)
NL = 10
NLT = NL + 1  # nlevels_tot
N_DAYS = 365

COST_TOL = 1.0e-14
GRAD_TOL = 1.0e-10
PARAM_TOL = 1.0e-9
RECOVERY_TOL = 1.0e-6
MAX_ITER = 200
MAX_LS_ITER = 20

W_MIN, W_MAX = 1.0e-4, 1.0
D_MIN, D_MAX = 1.0e-4, 10.0

LAMBDA_INIT, LAMBDA_MIN, LAMBDA_MAX = 1.0e-3, 1.0e-12, 1.0e12

HISTORY_FILE = "optimization_history.txt"
TRUE_PARAMS_FILE = "true_parameters.txt"
FLUXES_FILE = "fluxes_comparison.txt"


# Forward model wrappers
def build_model():
    model = Global_module_multilevel_matrix_eqx()
    # nlevels_tot must be a plain Python int so it stays static under jit
    # (it is used to build array shapes).
    return eqx.tree_at(lambda m: m.nlevels_tot, model, NLT)


MODEL = build_model()


def fup_top(w, d, mu, rs, t):
    """Upward flux at the top level for one (site, day). Pure and vmappable."""
    z = jnp.zeros(NLT, dtype=jnp.float64)
    # All of the fup, fdn, fab are that of shape (nlevels_tot,)
    _, _, fup, _ = MODEL.multilevel_matrix(NL, mu, rs, t, w, d, z, z, z)
    return fup[
        NLT - 1
    ]  # nlevels_tot is in Fortran value of 11, thus python is nlevels_tot - 1


# NOTE: vmap replaces the Fortran (site, day) double loop. It evaluates all
# n_sites*365 observations in one jit-compiled batched call (same results,
# far fewer Python-level calls), and lets jacfwd return the full (N, 2)
# Jacobian at once. w, d are shared (None); mu, rs, t vary per obs (0).
_batched = jax.vmap(fup_top, in_axes=(None, None, 0, 0, 0))


def forward(p, mu, rs, t):
    """p = [w, d]; mu, rs: (N,), t: (N, NLT) -> fup_top for all N obs."""
    return _batched(p[0], p[1], mu, rs, t)


fwd = jax.jit(forward)
# NOTE: # (N, 2) : [w, d] since mu, rs, t are fixed inputs
# and are not optimized so no derivatives with respect to them
# thus packing [w, d] together gives us (N, 2) matrix which allows for
# J.T @ J and J.T @ r
jac = jax.jit(jax.jacfwd(forward, argnums=0))


# Synthetic problem
def make_problem(n_sites, rng):
    """Mirror the Fortran setup. Arrays are flattened site-major: (n_sites*n_days)."""
    t_site = np.empty((n_sites, NLT))
    rs_site = np.empty(n_sites)
    mu_offset = np.empty(n_sites)
    mu_amp = np.empty(n_sites)

    for i in range(n_sites):
        isite = i + 1
        t_site[i] = 0.1 + (2.0 + isite - 0.1) * rng.random(NLT)
        rs_site[i] = rng.random()
        mu_offset[i] = 0.45 + 0.15 * rng.random()  # ~0.45-0.60
        mu_amp[i] = 0.20 + 0.20 * rng.random()  # ~0.20-0.40

    days = np.arange(1, N_DAYS + 1, dtype=float)
    mu = mu_offset[:, None] + mu_amp[:, None] * np.cos(
        2.0 * np.pi * (days[None, :] - 172.0) / 365.0
    )  # (n_sites, n_days)

    # NOTE: flatten to N = n_sites*n_days observations (site-major) so vmap can
    # map over axis 0 of mu, rs, t. Per-site rs and t are repeated N_DAYS times
    # to align with the flattened mu; the ordering must match across all three.
    mu_flat = mu.reshape(-1)  # (n_sites*n_days)
    rs_flat = np.repeat(rs_site, N_DAYS)  # rs_site = [a, b]  ->  [a, a, a, b, b, b]
    t_flat = np.repeat(t_site, N_DAYS, axis=0)  # (n_sites*n_days, NLT)

    w_true = max(rng.random(), W_MIN)
    d_true = 0.1 + (3.0 - 0.1) * rng.random()
    return (
        jnp.asarray(mu_flat),
        jnp.asarray(rs_flat),
        jnp.asarray(t_flat),
        float(w_true),
        float(d_true),
    )


# Levenberg-Marquardt
def calibrate(n_sites, seed=None):
    rng = np.random.default_rng(seed)
    mu, rs, t, w_true, d_true = make_problem(n_sites, rng)
    p_true = np.array([w_true, d_true])

    sep = "-" * 67
    print("\n" + sep)
    print(f" {n_sites}-SITE, FULL-YEAR ({N_DAYS} noon obs) CALIBRATION")
    print(sep)
    print(f"True parameters: w = {w_true:12.8f}, d = {d_true:12.8f}")

    with open(TRUE_PARAMS_FILE, "w") as f:
        f.write(f"True_w = {w_true:18.10E}\n")
        f.write(f"True_d = {d_true:18.10E}\n")
    print(f"True parameters written to: {TRUE_PARAMS_FILE}")

    # Synthetic references
    obs = fwd(jnp.asarray(p_true), mu, rs, t)
    print(f"Generated {n_sites * N_DAYS} synthetic references.")

    # Initial guess
    p = np.clip(np.array([0.5, 0.5]), [W_MIN, D_MIN], [W_MAX, D_MAX])
    lam = LAMBDA_INIT
    lo, hi = np.array([W_MIN, D_MIN]), np.array([W_MAX, D_MAX])

    print("\n" + sep)
    print(f" LEVENBERG-MARQUARDT OPTIMIZATION ({n_sites} sites x {N_DAYS} days)")
    print(sep)
    print(f"Initial guess: w = {p[0]:12.8f}, d = {p[1]:12.8f}\n")

    converged = False
    history = []
    H = np.zeros((2, 2))
    cost = np.nan

    for it in range(1, MAX_ITER + 1):
        pj = jnp.asarray(p)
        r = np.asarray(fwd(pj, mu, rs, t) - obs)
        J = np.asarray(jac(pj, mu, rs, t))  # (n_sites*n_days, 2)

        cost = 0.5 * float(r @ r)
        g = J.T @ r  # shape: 2 x 2
        H = J.T @ J  # J.T @ J is symmetric thus h_wd = h_dw
        grad_norm = float(np.linalg.norm(g))

        h_ww, h_wd, h_dd = H[0, 0], H[0, 1], H[1, 1]
        jac_corr = (
            h_wd / np.sqrt(h_ww * h_dd) if (h_ww > 1e-30 and h_dd > 1e-30) else 0.0
        )

        if cost < COST_TOL:
            print("Converged: cost tolerance reached.")
            converged = True
            break
        if grad_norm < GRAD_TOL:
            print("Converged: gradient tolerance reached.")
            converged = True
            break

        # Damped Gauss-Newton 2x2 system
        det = (h_ww + lam) * (h_dd + lam) - h_wd**2
        if abs(det) < 1e-30:
            lam = min(10.0 * lam, LAMBDA_MAX)
            print(f"Near-singular Hessian. Increasing lambda to {lam:12.4E}")
            continue

        dp = np.array(
            [
                (-(h_dd + lam) * g[0] + h_wd * g[1]) / det,
                (h_wd * g[0] - (h_ww + lam) * g[1]) / det,
            ]
        )
        delta_norm = float(np.linalg.norm(dp))

        if delta_norm < PARAM_TOL * (1.0 + np.linalg.norm(p)):
            if grad_norm < GRAD_TOL:
                print("Converged: small parameter step and gradient.")
                converged = True
                break
            lam = min(10.0 * lam, LAMBDA_MAX)
            continue

        # Backtracking line search (plain forward model, all sites)
        alpha, accepted = 1.0, False
        p_trial, cost_new = p, cost
        for _ in range(MAX_LS_ITER):
            p_trial = np.clip(p + alpha * dp, lo, hi)
            r_t = np.asarray(fwd(jnp.asarray(p_trial), mu, rs, t) - obs)
            cost_new = 0.5 * float(r_t @ r_t)
            if cost_new < cost:
                accepted = True
                break
            alpha *= 0.5

        if accepted:
            step = alpha * dp
            actual = cost - cost_new
            predicted = -(g @ step) - 0.5 * (step @ H @ step)
            rho = actual / predicted if predicted > 1e-30 else 0.0

            p, cost = p_trial, cost_new

            if rho > 0.75:
                lam = max(lam / 3.0, LAMBDA_MIN)
            elif rho < 0.25:
                lam = min(lam * 4.0, LAMBDA_MAX)
            if alpha < 1.0:
                lam = min(lam * 2.0, LAMBDA_MAX)
        else:
            lam = min(lam * 10.0, LAMBDA_MAX)
            rho = -1.0

        print(
            f"iter={it:4d} cost={cost:12.4E} w={p[0]:10.6f} d={p[1]:10.6f} "
            f"lambda={lam:10.3E} alpha={alpha:10.3E} rho={rho:8.4f}"
        )

        history.append(
            (it, cost, p[0], p[1], g[0], g[1], lam, rho if accepted else -1.0)
        )

        if np.all(np.abs(p - p_true) < RECOVERY_TOL):
            print("Recovered the true parameters.")
            converged = True
            break

        if it % 20 == 0:
            print(f"Jacobian correlation = {jac_corr:12.4E}")
            if abs(jac_corr) > 0.999:
                print("WARNING: w and d may be poorly identifiable.")

    # Results
    print("\n" + sep)
    print(" OPTIMIZATION COMPLETE")
    print(sep)
    print("Status: CONVERGED" if converged else "Status: MAXIMUM ITERATIONS REACHED")
    print(f"True w       = {w_true:12.8f}")
    print(f"Recovered w  = {p[0]:12.8f}")
    print(f"Absolute error w = {abs(p[0] - w_true):14.6E}")
    print(f"True d       = {d_true:12.8f}")
    print(f"Recovered d  = {p[1]:12.8f}")
    print(f"Absolute error d = {abs(p[1] - d_true):14.6E}")
    print(f"Final cost = {cost:18.10E}")

    det = H[0, 0] * H[1, 1] - H[0, 1] ** 2
    if det > 1e-30:
        print(f"Approx. std. error on w = {np.sqrt(H[1, 1] / det):10.6f}")
        print(f"Approx. std. error on d = {np.sqrt(H[0, 0] / det):10.6f}")

    with open(HISTORY_FILE, "w") as f:
        f.write("# Optimization History\n")
        f.write("# Columns: iter cost w d gJ_w gJ_d lambda rho\n")
        for row in history:
            f.write(f"{row[0]:6d}" + "".join(f"{v:18.10E}" for v in row[1:]) + "\n")
    print(f"Optimization history written to: {HISTORY_FILE}")

    # We reshape it back to (n_sites, N_DAYS)
    fup_model = np.asarray(fwd(jnp.asarray(p), mu, rs, t)).reshape(n_sites, N_DAYS)
    fup_ref = np.asarray(obs).reshape(n_sites, N_DAYS)
    with open(FLUXES_FILE, "w") as f:
        f.write("# Flux Comparison (Timeseries, multi-site)\n")
        f.write("# Columns: site day fup_model fup_obs\n")
        for i in range(n_sites):
            for j in range(N_DAYS):
                f.write(
                    f"{i + 1:6d}{j + 1:6d}{fup_model[i, j]:18.10E}{fup_ref[i, j]:18.10E}\n"
                )
    print(f"Flux comparison written to: {FLUXES_FILE}")

    return p, p_true, converged


def main():
    ap = argparse.ArgumentParser(description="JAX multi-site LM calibration of (w, d)")
    ap.add_argument("--n_sites", type=int, default=8)
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (default: random, like the Fortran /dev/urandom seed)",
    )
    args = ap.parse_args()
    calibrate(args.n_sites, args.seed)


if __name__ == "__main__":
    main()
