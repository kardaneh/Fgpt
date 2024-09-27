def hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag):
    froz_frac_moy = np.zeros((kjpindex),dtype=float32)
    smtot_moy = np.zeros((kjpindex),dtype=float32)
    mc_ns = np.zeros((nslm, kjpindex),dtype=float32)
    for jsl in range(0, nslm, 1):
        for ji in range(0, kjpindex, 1):
            m = 1. - 1. / nvan[ji]
            if ((not ok_thermodynamical_freezing) or (mc[ji, jsl, ins] < (mcr[ji] + min_sechiba))) :
                if (stempdiag[ji, jsl] >= (fr_center + fr_dT / 2.)) :
                    x = 1.
                elif ((stempdiag[ji, jsl] >= (fr_center - fr_dT / 2.)) and (stempdiag[ji, jsl] < (fr_center + fr_dT / 2.))) :
                    x = (stempdiag[ji, jsl] - (fr_center - fr_dT / 2.)) / fr_dT
                else:
                    x = 0.
            elif (ok_thermodynamical_freezing) :
                if (stempdiag[ji, jsl] >= (fr_center + fr_dT / 2.)) :
                    x = 1.
                elif ((stempdiag[ji, jsl] >= (fr_center - fr_dT / 2.)) and (stempdiag[ji, jsl] < (fr_center + fr_dT / 2.))) :
                    x = min(((mcs[ji] - mcr[ji]) * ((2.2 * 1000. * avan[ji] * (fr_center + fr_dT / 2. - stempdiag[ji, jsl]) * lhf / ZeroCelsius / 10.) ** nvan[ji] + 1.) ** (- m)) / (mc[ji, jsl, ins] - mcr[ji]), 1.)
                else:
                    x = 0.
            profil_froz_hydro_ns[ji, jsl, ins] = 1. - x
            mc_ns[ji, jsl] = mc[ji, jsl, ins] / mcs[ji]
    froz_frac_moy[:] = zero
    denom = zero
    for jsl in range(0, nslm, 1):
        froz_frac_moy[:] = froz_frac_moy[:] + dh[jsl] * profil_froz_hydro_ns[:, jsl, ins]
        denom = denom + dh[jsl]
    froz_frac_moy[:] = froz_frac_moy[:] / denom
    smtot_moy[:] = zero
    denom = zero
    for jsl in range(0, nslm - 1, 1):
        smtot_moy[:] = smtot_moy[:] + dh[jsl] * mc_ns[:, jsl]
        denom = denom + dh[jsl]
    smtot_moy[:] = smtot_moy[:] / denom
    for jsl in range(0, nslm, 1):
        profil_froz_hydro_ns[:, jsl, ins] = min(profil_froz_hydro_ns[:, jsl, ins] * (froz_frac_moy[:] ** froz_frac_corr) * (smtot_moy[:] ** smtot_corr), max_froz_hydro)

