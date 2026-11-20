#!/usr/bin/env python3

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from scipy.io import FortranFile


class Global_module_multilevel_matrix_eqx(eqx.Module):
    nice: int
    ncirc: int
    nsnow: int
    nslm: int
    nvm: int
    nstm: int
    kjpindex: int
    ier: int
    ic0: int
    ic: int
    icr: float
    start_time: float
    stop_time: float
    nlevels_tot: int

    def __init__(self):
        self.nice = jnp.int32(8)
        self.ncirc = jnp.int32(3)
        self.nsnow = jnp.int32(3)
        self.nslm = jnp.int32(11)
        self.nvm = jnp.int32(15)
        self.nstm = jnp.int32(3)
        self.kjpindex = jnp.int32(4717)
        self.ier = jnp.int32(0)
        self.ic0 = jnp.int32(0)
        self.ic = jnp.int32(0)
        self.icr = jnp.float64(0.0)
        self.start_time = jnp.float64(0.0)
        self.stop_time = jnp.float64(0.0)
        self.nlevels_tot = jnp.int32(0)

    def declaration_initialization(self):
        print("--- add the declaration and initialization in module global ---")
        path = "benchmark/multilevel_matrix/global.bin"
        ffile = FortranFile(path, "r")
        nlevels_tot = int(ffile.read_ints(jnp.int32)[0])
        updated = {"nlevels_tot": nlevels_tot}
        return eqx.tree_at(
            lambda m: tuple(getattr(m, k) for k in updated.keys()),
            self,
            tuple(updated.values()),
            is_leaf=lambda x: x is None,
        )

    def trdiag(self, n, a, b, c, g, x):
        def _scan_body_0(carry, i):
            (a, b, c, g) = carry
            t = a[i] / b[i - 1]
            b = b.at[i].subtract(c[i - 1] * t)
            g = g.at[i].subtract(g[i - 1] * t)
            carry = (a, b, c, g)
            return (carry, None)

        ((a, b, c, g), _) = lax.scan(_scan_body_0, (a, b, c, g), jnp.arange(1, n))
        x = x.at[n - 1].set(g[n - 1] / b[n - 1])

        def _scan_body_1(carry, i):
            (b, c, g, x) = carry
            x = x.at[i].set((g[i] - c[i] * x[i + 1]) / b[i])
            carry = (b, c, g, x)
            return (carry, None)

        ((b, c, g, x), _) = lax.scan(
            _scan_body_1, (b, c, g, x), jnp.arange(n - 2, -1, -1)
        )
        return (g, x, b)

    @eqx.filter_jit
    def multilevel_matrix(self, nl, mu, rs, t, w, d, fup, fdn, fab):
        xn = jnp.zeros((nl * 2,), dtype=jnp.float64)
        an = jnp.zeros((nl * 2,), dtype=jnp.float64)
        bn = jnp.zeros((nl * 2,), dtype=jnp.float64)
        cn = jnp.zeros((nl * 2,), dtype=jnp.float64)
        gn = jnp.zeros((nl * 2,), dtype=jnp.float64)
        data_pu = jnp.zeros((2, self.nlevels_tot), dtype=jnp.float64)
        data_nl = jnp.zeros((11, nl), dtype=jnp.float64)
        gam = jnp.zeros((4,), dtype=jnp.float64)
        g = jnp.float64(0.5)
        # mubar = jnp.float64(1.0)
        fab = fab.at[:].set(0.0)
        k = g / mu
        beta0 = 0.5 / w * (w + 2.0 * d * mu / 3.0)
        beta = 0.5 / w * (w + d / 3.0)
        gam = gam.at[0].set(1.0 - (1.0 - beta) * w)
        gam = gam.at[1].set(w * beta)
        gam = gam.at[2].set(beta0)
        gam = gam.at[3].set(1.0 - gam[2])
        mu0 = 1.0 / k
        fspi = 1.0 / mu0
        data_nl = data_nl.at[0, :].set(t[nl - 1 :: -1])
        lambda_ = jnp.sqrt(gam[0] ** 2 - gam[1] ** 2)
        cap_gam = gam[1] / (gam[0] + lambda_)
        data_pu = data_pu.at[0, 0].set(0.0)

        def _scan_body_0(carry, lev):
            (data_nl, data_pu) = carry
            data_pu = data_pu.at[0, lev].set(data_pu[0, lev - 1] + data_nl[0, lev - 1])
            carry = (data_nl, data_pu)
            return (carry, None)

        ((data_nl, data_pu), _) = lax.scan(
            _scan_body_0, (data_nl, data_pu), jnp.arange(1, self.nlevels_tot)
        )
        fdb = 0.0
        data_pu = data_pu.at[1, 0].set(mu0 * fspi)

        def _scan_body_1(carry, lev):
            (cap_gam, data_nl, data_pu, fspi, gam, lambda_, mu0, w) = carry
            exp_trm = jnp.exp(-jnp.minimum(lambda_ * data_nl[0, lev], 30.0))
            data_nl = data_nl.at[1, lev].set(1.0 + cap_gam * exp_trm)
            data_nl = data_nl.at[2, lev].set(1.0 - cap_gam * exp_trm)
            data_nl = data_nl.at[3, lev].set(cap_gam + exp_trm)
            data_nl = data_nl.at[4, lev].set(cap_gam - exp_trm)
            cpn = w * fspi * ((gam[0] - 1.0 / mu0) * gam[2] + gam[3] * gam[1])
            cmn = w * fspi * ((gam[0] + 1.0 / mu0) * gam[3] + gam[1] * gam[2])
            den = lambda_**2 - 1.0 / mu0**2
            exp_top = jnp.exp(-jnp.minimum(data_pu[0, lev] / mu0, 30.0))
            exp_bot = exp_top * jnp.exp(-jnp.minimum(data_nl[0, lev] / mu0, 30.0))
            data_pu = data_pu.at[1, lev + 1].set(mu0 * fspi * exp_bot)
            data_nl = data_nl.at[5, lev].set(exp_top * cpn / den)
            data_nl = data_nl.at[6, lev].set(exp_bot * cpn / den)
            data_nl = data_nl.at[7, lev].set(exp_top * cmn / den)
            data_nl = data_nl.at[8, lev].set(exp_bot * cmn / den)
            carry = (cap_gam, data_nl, data_pu, fspi, gam, lambda_, mu0, w)
            return (carry, None)

        ((cap_gam, data_nl, data_pu, fspi, gam, lambda_, mu0, w), _) = lax.scan(
            _scan_body_1,
            (cap_gam, data_nl, data_pu, fspi, gam, lambda_, mu0, w),
            jnp.arange(nl),
        )
        bs = rs * data_pu[1, self.nlevels_tot - 1]
        an = an.at[0].set(0.0)
        bn = bn.at[0].set(data_nl[1, 0])
        cn = cn.at[0].set(-data_nl[2, 0])
        gn = gn.at[0].set(fdb - data_nl[7, 0])

        def _scan_body_2(carry, lev):
            (an, bn, cn, data_nl, gn) = carry
            le = 2 * lev + 1
            lo = 2 * lev + 2
            an = an.at[le].set(
                data_nl[2, lev + 1] * data_nl[1, lev]
                - data_nl[4, lev + 1] * data_nl[3, lev]
            )
            an = an.at[lo].set(
                data_nl[2, lev] * data_nl[3, lev] - data_nl[4, lev] * data_nl[1, lev]
            )
            bn = bn.at[le].set(
                data_nl[2, lev] * data_nl[2, lev + 1]
                - data_nl[4, lev] * data_nl[4, lev + 1]
            )
            bn = bn.at[lo].set(
                data_nl[1, lev] * data_nl[1, lev + 1]
                - data_nl[3, lev] * data_nl[3, lev + 1]
            )
            cn = cn.at[le].set(
                data_nl[1, lev + 1] * data_nl[4, lev + 1]
                - data_nl[2, lev + 1] * data_nl[3, lev + 1]
            )
            cn = cn.at[lo].set(
                data_nl[3, lev] * data_nl[4, lev + 1]
                - data_nl[1, lev] * data_nl[2, lev + 1]
            )
            gn = gn.at[le].set(
                data_nl[2, lev + 1] * (data_nl[5, lev + 1] - data_nl[6, lev])
                - data_nl[4, lev + 1] * (data_nl[7, lev + 1] - data_nl[8, lev])
            )
            gn = gn.at[lo].set(
                data_nl[3, lev] * (data_nl[5, lev + 1] - data_nl[6, lev])
                + data_nl[1, lev] * (data_nl[8, lev] - data_nl[7, lev + 1])
            )
            carry = (an, bn, cn, data_nl, gn)
            return (carry, None)

        ((an, bn, cn, data_nl, gn), _) = lax.scan(
            _scan_body_2, (an, bn, cn, data_nl, gn), jnp.arange(nl - 1)
        )
        lev = 2 * nl
        an = an.at[lev - 1].set(data_nl[1, nl - 1] - rs * data_nl[3, nl - 1])
        bn = bn.at[lev - 1].set(data_nl[2, nl - 1] - rs * data_nl[4, nl - 1])
        cn = cn.at[lev - 1].set(0.0)
        gn = gn.at[lev - 1].set(bs - data_nl[6, nl - 1] + rs * data_nl[8, nl - 1])
        (gn, xn, bn) = self.trdiag(lev, an, bn, cn, gn, xn)
        data_nl = data_nl.at[9, :].set(xn[0:lev:2])
        data_nl = data_nl.at[10, :].set(xn[1:lev:2])
        fdu = (
            data_nl[9, 0] * data_nl[3, 0]
            - data_nl[10, 0] * data_nl[4, 0]
            + data_nl[5, 0]
        )
        fup = fup.at[self.nlevels_tot - 1].set(fdu)
        fdn = fdn.at[self.nlevels_tot - 1].set(fdb + data_pu[1, 0])

        def _scan_body_3(carry, lev):
            (data_nl, data_pu, fdn, fup) = carry
            le = self.nlevels_tot - 2 - lev
            fdu = (
                data_nl[9, lev] * data_nl[1, lev]
                + data_nl[10, lev] * data_nl[2, lev]
                + data_nl[6, lev]
            )
            fup = fup.at[le].set(fdu)
            fdb = (
                data_nl[9, lev] * data_nl[3, lev]
                + data_nl[10, lev] * data_nl[4, lev]
                + data_nl[8, lev]
            )
            fdn = fdn.at[le].set(fdb + data_pu[1, lev + 1])
            carry = (data_nl, data_pu, fdn, fup)
            return (carry, None)

        ((data_nl, data_pu, fdn, fup), _) = lax.scan(
            _scan_body_3, (data_nl, data_pu, fdn, fup), jnp.arange(nl)
        )
        fab = fab.at[nl - 1 :: -1].set(
            fdn[self.nlevels_tot - 1 : 2 - 2 : -1]
            - fup[self.nlevels_tot - 1 : 2 - 2 : -1]
            - (fdn[nl - 1 :: -1] - fup[nl - 1 :: -1])
        )
        return (fab, fdn, fup, eqx.tree_at(lambda m: (), self, ()))
