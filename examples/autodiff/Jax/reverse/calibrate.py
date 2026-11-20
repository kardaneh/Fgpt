#!/usr/bin/env python3

# Copyright 2026 CNRS
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/
"""
Joint multi-site calibration of w, d (shared) and rs (one per site) for the
two-stream radiative transfer model, using the JAX forward model and
Levenberg-Marquardt.

Python/JAX port of calibrate_w_d_rs.F90 (adjoint version, twin experiment):
  - parameter vector p = [w, d, rs(1), ..., rs(n_sites)]   (np = n_sites + 2)
  - the Fortran adjoint call per (site, day), seeded with fupb(top) = 1,
    becomes jax.grad of a scalar per-observation function, vmapped over all
    observations (reverse mode, like Tapenade's multilevel_matrix_b)
  - each observation's gradient w.r.t. (w, d, rs_site) is scattered into one
    row of the (N, np) Jacobian, exactly like row_grad in the Fortran code

Usage:
    python calibrate_w_d_rs.py --n_sites 8 [--seed 0]
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from global_module_mutlilevel_matrix_b import Global_module_multilevel_matrix_eqx

jax.config.update("jax_enable_x64", True)


# Settings (same as the Fortran program)
NL = 10
NLT = NL + 1  # nlevels_tot
N_DAYS = 365

COST_TOL = 1.0e-16
PARAM_TOL = 1.0e-10
MAX_ITER = 200
MAX_LS_ITER = 100

LAMBDA_INIT, LAMBDA_MIN, LAMBDA_MAX = 1.0e-3, 1.0e-12, 1.0e12

TRUE_FILE = "w_d_rs_true.txt"
HISTORY_FILE = "w_d_rs_optimization_history.txt"
PROFILE_FILE = "w_d_rs_recovered.txt"
FLUX_FILE = "w_d_rs_flux_comparison.txt"


# Forward model and derivatives
def build_model():
    model = Global_module_multilevel_matrix_eqx()
    # nlevels_tot must be a plain Python int so it stays static under jit
    return eqx.tree_at(lambda m: m.nlevels_tot, model, NLT)


MODEL = build_model()


def fup_top(w, d, rs, mu, t):
    """Upward flux at the top level for one (site, day). Pure and vmappable."""
    z = jnp.zeros(NLT, dtype=jnp.float64)
    _, _, fup, _ = MODEL.multilevel_matrix(NL, mu, rs, t, w, d, z, z, z)
    return fup[NLT - 1]


# w, d shared (None); rs, mu, t vary per observation (0)
_batched = jax.vmap(fup_top, in_axes=(None, None, 0, 0, 0))

# NOTE: Reverse mode (adjoint), like the Fortran multilevel_matrix_b call.
# fup_top returns one scalar per observation, so jax.grad gives its gradient
# w.r.t. (w, d, rs) in one backward pass, and vmap does this for all
# observations at once. Each observation depends only on w, d and its own
# site's rs, so these 3 entries are scattered into the right columns of the
# Jacobian row (all other entries are zero). jax.jacfwd would give the same
# Jacobian but the choice just mirrors the Fortran version.
_batched_grad = jax.vmap(
    jax.grad(fup_top, argnums=(0, 1, 2)), in_axes=(None, None, 0, 0, 0)
)


def forward(p, mu, t, site_idx):
    """p = [w, d, rs_1..rs_n]; returns fup_top for all N observations."""
    return _batched(p[0], p[1], p[2 + site_idx], mu, t)


def jacobian(p, mu, t, site_idx):
    """(N, np) Jacobian. Row k has entries only in columns 0, 1 and 2+site(k)."""
    gw, gd, grs = _batched_grad(p[0], p[1], p[2 + site_idx], mu, t)
    n = mu.shape[0]
    J = jnp.zeros((n, p.shape[0]), dtype=jnp.float64)
    J = J.at[:, 0].set(gw)
    J = J.at[:, 1].set(gd)
    J = J.at[jnp.arange(n), 2 + site_idx].set(grs)
    return J


fwd = jax.jit(forward)
jac = jax.jit(jacobian)


# Synthetic problem
def make_problem(n_sites, rng):
    """Mirror the Fortran setup. Arrays are flattened site-major."""
    t_site = 0.2 + 1.8 * rng.random((n_sites, NLT))
    rs_true = 0.05 + 0.35 * rng.random(n_sites)
    mu_offset = 0.45 + 0.35 * rng.random(n_sites)
    mu_amp = 0.5 * mu_offset * (0.4 + 0.4 * rng.random(n_sites))

    days = np.arange(1, N_DAYS + 1, dtype=float)
    mu = mu_offset[:, None] + mu_amp[:, None] * np.cos(
        2.0 * np.pi * (days[None, :] - 172.0) / 365.0
    )
    mu = np.maximum(0.05, mu)  # (n_sites, n_days)

    w_true = 0.5 + 0.45 * rng.random()  # (0.5, 0.95)
    d_true = 0.5 + 2.0 * rng.random()  # (0.5, 2.5)

    # NOTE: flatten to N = n_sites*n_days observations (site-major) so vmap can
    # map over axis 0 of mu, rs, t. Per-site t is repeated N_DAYS times, and
    # site_idx tells each observation which rs parameter it uses.
    mu_flat = mu.reshape(-1)
    t_flat = np.repeat(t_site, N_DAYS, axis=0)  # (N, NLT)
    site_idx = np.repeat(np.arange(n_sites), N_DAYS)  # (N,)

    return (
        jnp.asarray(mu_flat),
        jnp.asarray(t_flat),
        jnp.asarray(site_idx),
        float(w_true),
        float(d_true),
        rs_true,
    )


# Levenberg-Marquardt
def calibrate(n_sites, seed=None):
    rng = np.random.default_rng(seed)
    mu, t, site_idx, w_true, d_true, rs_true = make_problem(n_sites, rng)
    n_obs = n_sites * N_DAYS
    n_par = n_sites + 2

    p_true = np.concatenate([[w_true, d_true], rs_true])
    p_min = np.concatenate([[0.001, 0.001], np.zeros(n_sites)])
    p_max = np.concatenate([[0.999, 10.0], np.full(n_sites, 0.99)])

    print("\n\n JOINT w, d, rs(site) CALIBRATION (ADJOINT + LEVENBERG-MARQUARDT)\n")
    print(f"Number of sites          = {n_sites}")
    print(f"Number of days           = {N_DAYS}")
    print(f"Number of parameters     = {n_par}")
    print(f"True w                   = {w_true:12.6f}")
    print(f"True d                   = {d_true:12.6f}\n")

    with open(TRUE_FILE, "w") as f:
        f.write(f"True_w = {w_true:18.10E}\n")
        f.write(f"True_d = {d_true:18.10E}\n")
        for i in range(n_sites):
            f.write(f"True_rs({i + 1}) = {rs_true[i]:18.10E}\n")

    # Synthetic references from the TRUE parameters
    print("Generating synthetic references...")
    obs = fwd(jnp.asarray(p_true), mu, t, site_idx)
    print(f"Generated {n_obs} synthetic references.")

    # Initial guess (deliberately different from the truth)
    p = np.clip(np.full(n_par, 0.5), p_min, p_max)
    print(f"\nInitial guess: w = {p[0]:10.6f}  d = {p[1]:10.6f}")
    print(f"Initial guess: rs (all sites) = {p[2]:10.6f}")

    # Gradient tolerance from the residual-weighted gradient at (site 1, day 1)
    r0 = np.asarray(fwd(jnp.asarray(p), mu, t, site_idx) - obs)
    J0 = np.asarray(jac(jnp.asarray(p), mu, t, site_idx))
    test_grad = r0[0] * J0[0]
    gradient_tolerance = max(1.0e-14, 1.0e-8 * np.max(np.abs(test_grad)))
    print(
        f"\nMax |gradient| at initial point (site 1) = {np.max(np.abs(test_grad)):12.4E}"
    )
    print(f"Gradient tolerance                       = {gradient_tolerance:12.4E}")

    lam = LAMBDA_INIT
    converged = False
    history = []

    cost = 0.5 * float(r0 @ r0)
    print(f"\nInitial cost = {cost:18.10E}")

    for it in range(1, MAX_ITER + 1):
        old_cost = cost
        pj = jnp.asarray(p)

        r = np.asarray(fwd(pj, mu, t, site_idx) - obs)
        J = np.asarray(jac(pj, mu, t, site_idx))  # (N, np)

        cost = 0.5 * float(r @ r)
        grad = J.T @ r  # J^T r
        H = J.T @ J  # J^T J
        grad_norm = float(np.linalg.norm(grad))
        max_gradient = float(np.max(np.abs(grad)))

        if grad_norm < gradient_tolerance:
            print("Converged: gradient tolerance reached.")
            converged = True
            # NOTE: This final row is a bookkeeping entry, not a real iteration.
            # No step was taken: the cost and lambda are unchanged from the
            # previous row and step_norm is set to 0.0. In the plots this shows
            # up as a flat cost, a step-norm point that drops off the log axis,
            # and Cost/Cost_prev = 1. It mirrors the Fortran version, which
            # writes the same kind of last row. It is not an optimizer stall.
            history.append((it, cost, grad_norm, lam, 0.0, max_gradient))
            break
        if cost < COST_TOL:
            print("Converged: cost tolerance reached.")
            converged = True
            break

        # Damped Gauss-Newton step, increase lambda until the step reduces cost
        accepted = False
        p_trial, cost_new, delta = p, cost, np.zeros(n_par)
        for _ in range(MAX_LS_ITER):
            diag = np.diag(H)
            H_lm = H + np.diag(lam * np.maximum(diag, 1.0e-30))  # Marquardt scaling

            # NOTE: np.linalg.solve replaces solve_linear_system; LinAlgError plays the role
            # of the Fortran singular flag (raise lambda 10x and retry).
            try:
                delta = np.linalg.solve(
                    H_lm, -grad
                )  # the solve_linear_system is replaced with np.linalg.solve
            except np.linalg.LinAlgError:
                lam = min(10.0 * lam, LAMBDA_MAX)
                continue

            p_trial = np.clip(p + delta, p_min, p_max)
            r_t = np.asarray(fwd(jnp.asarray(p_trial), mu, t, site_idx) - obs)
            cost_new = 0.5 * float(r_t @ r_t)

            if cost_new < cost:
                accepted = True
                break
            lam = min(10.0 * lam, LAMBDA_MAX)

        if accepted:
            actual = cost - cost_new
            predicted = -(grad @ delta) - 0.5 * (delta @ H @ delta)
            rho = actual / predicted if predicted > 1e-30 else 0.0

            step_norm = float(np.linalg.norm(p_trial - p))
            p, cost = p_trial, cost_new

            if rho > 0.75:
                lam = max(lam / 3.0, LAMBDA_MIN)
            elif rho < 0.25:
                lam = min(lam * 4.0, LAMBDA_MAX)
        else:
            print(f"LM step failed at iteration {it:6d}.")
            step_norm = 0.0

        history.append((it, cost, grad_norm, lam, step_norm, max_gradient))
        print(
            f"iter={it:5d} cost={cost:14.6E} grad={grad_norm:12.4E} "
            f"lambda={lam:10.3E} step={step_norm:12.4E}  "
            f"w={p[0]:10.6f} d={p[1]:10.6f}"
        )

        if step_norm < PARAM_TOL * (1.0 + np.linalg.norm(p)):
            print("Converged: parameter step tolerance reached.")
            converged = True
            break
        if abs(old_cost - cost) < COST_TOL * max(1.0, old_cost):
            print("Converged: cost change tolerance reached.")
            converged = True
            break

    # Results
    print("\n\n CALIBRATION COMPLETE\n")
    print("Status: CONVERGED" if converged else "Status: MAXIMUM ITERATIONS REACHED")
    print(f"Final cost = {cost:18.10E}\n")
    print(
        f"w:  true={w_true:12.8f}  recovered={p[0]:12.8f}  err={abs(p[0] - w_true):14.6E}"
    )
    print(
        f"d:  true={d_true:12.8f}  recovered={p[1]:12.8f}  err={abs(p[1] - d_true):14.6E}\n"
    )
    for i in range(n_sites):
        print(
            f"rs({i + 1:3d}): true={rs_true[i]:10.6f}  recovered={p[2 + i]:10.6f}  "
            f"err={abs(p[2 + i] - rs_true[i]):14.6E}"
        )

    with open(HISTORY_FILE, "w") as f:
        f.write("# w,d,rs adjoint Levenberg-Marquardt calibration history\n")
        f.write("# Columns: iter cost grad_norm lambda_lm step_norm max_gradient\n")
        for row in history:
            f.write(f"{row[0]:8d}" + "".join(f"{v:20.10E}" for v in row[1:]) + "\n")

    with open(PROFILE_FILE, "w") as f:
        f.write("# Recovered w, d, rs(site)\n")
        f.write(f"w_d : {p[0]:20.12E}{w_true:20.12E}{abs(p[0] - w_true):20.12E}\n")
        f.write(f"d   : {p[1]:20.12E}{d_true:20.12E}{abs(p[1] - d_true):20.12E}\n")
        for i in range(n_sites):
            f.write(
                f"rs({i + 1}) : {p[2 + i]:20.12E}{rs_true[i]:20.12E}"
                f"{abs(p[2 + i] - rs_true[i]):20.12E}\n"
            )

    fup_model = np.asarray(fwd(jnp.asarray(p), mu, t, site_idx)).reshape(
        n_sites, N_DAYS
    )
    fup_ref = np.asarray(obs).reshape(n_sites, N_DAYS)
    with open(FLUX_FILE, "w") as f:
        f.write("# Final flux comparison\n")
        f.write("# Columns: site day Fup_model Fup_obs residual\n")
        for i in range(n_sites):
            for j in range(N_DAYS):
                m, o = fup_model[i, j], fup_ref[i, j]
                f.write(f"{i + 1:8d}{j + 1:8d}{m:20.12E}{o:20.12E}{m - o:20.12E}\n")

    print(f"\nTrue parameters written to: {TRUE_FILE}")
    print(f"Optimization history written to: {HISTORY_FILE}")
    print(f"Recovered parameters written to: {PROFILE_FILE}")
    print(f"Flux comparison written to: {FLUX_FILE}")

    return p, p_true, converged


def main():
    ap = argparse.ArgumentParser(
        description="JAX multi-site LM calibration of (w, d, rs)"
    )
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
