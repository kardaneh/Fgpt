
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: altmax
  REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ncirc, nparts, nelements) :: circ_class_biomass
  REAL(KIND = r_std), DIMENSION(kjpindex) :: drainage
  REAL(KIND = r_std), DIMENSION(kjpindex) :: drysoil_frac
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm) :: e_frac
  REAL(KIND = r_std), DIMENSION(kjpindex) :: evap_bare_lim, evap_bare_lim_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: evap_bare_lim_ns, evap_bare_lim_ns_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot
  REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot_penm
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: F_absorption
  REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: humrel
  REAL(KIND = r_std), DIMENSION(kjpindex) :: irrigation
  REAL(KIND = r_std), DIMENSION(kjpindex) :: k_litt
  REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: ksoil, ksoil_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: litterhumdiag
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mc_layh, mc_layh_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mc_layh_s, mc_layh_s_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mcl_layh, mcl_layh_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mcl_layh_s, mcl_layh_s_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
  REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: reinf_slope_soil
  REAL(KIND = r_std), DIMENSION(kjpindex) :: reinfiltration
  REAL(KIND = r_std), DIMENSION(kjpindex) :: returnflow
  REAL(KIND = r_std), DIMENSION(kjpindex) :: root_deficit, root_deficit_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ndepths) :: root_depth
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof) :: root_profile
  REAL(KIND = r_std), DIMENSION(kjpindex) :: runoff
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: shumdiag
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: shumdiag_perma
  REAL(KIND = r_std), DIMENSION(kjpindex) :: snow
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
  REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_bare_soil
  REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_melt
  REAL(KIND = r_std), DIMENSION(kjpindex) :: totfrac_nobio
  REAL(KIND = r_std), DIMENSION(kjpindex) :: tq_cdrag
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: transpir
  REAL(KIND = r_std), DIMENSION(kjpindex) :: u
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us, us_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: v
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: vegstress
  REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu
  INTEGER(KIND = i_std) :: ji
  INTEGER(KIND = i_std) :: error_flag_hydrol_diag_soil_flux_1
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_1
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_2
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_3
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_4
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_5
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_6
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_7
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_8
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_9
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_10
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_11
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_12
  INTEGER(KIND = i_std) :: error_flag_hydrol_soil_infilt_1
  INTEGER(KIND = i_std) :: error_flag_hydrol_root_profile_1
  INTEGER(KIND = i_std) :: error_flag_hydrol_root_profile_2
  WRITE(*, *) '--- inside the main program ---'
  CALL declaration_initialization
  CALL read_dummy(altmax, avan, circ_class_biomass, e_frac, evap_bare_lim, evap_bare_lim_ns, evapot, evapot_penm, F_absorption, &
&frac_snow_nobio, humrel, irrigation, ks, mcfc, mcr, mcs, mcw, njsc, nvan, precip_rain, reinf_slope_soil, reinfiltration, &
&returnflow, snow, snowdz, soiltile, stempdiag, tot_bare_soil, tot_melt, totfrac_nobio, tq_cdrag, transpir, u, us, v, veget, &
&veget_max, vevapnu)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  CALL hydrol_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget, veget_max, soiltile, njsc, reinf_slope_soil, transpir, &
&vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, reinfiltration, irrigation, tot_melt, evap_bare_lim, &
&evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, stempdiag, snow, snowdz, &
&tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, e_frac, ksoil, altmax, root_profile, root_depth, &
&root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, frac_snow_nobio, F_absorption)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  mc_cpu = mc
  dr_ns_cpu = dr_ns
  mcl_cpu = mcl
  qflux_ns_cpu = qflux_ns
  soil_wet_ns_cpu = soil_wet_ns
  tmc_litter_cpu = tmc_litter
  vegstressv_cpu = vegstressv
  soil_wet_litter_cpu = soil_wet_litter
  tmc_cpu = tmc
  ae_ns_cpu = ae_ns
  ru_ns_cpu = ru_ns
  humrelv_cpu = humrelv
  profil_froz_hydro_ns_cpu = profil_froz_hydro_ns
  rhs_cpu = rhs
  tmat_cpu = tmat
  resolv_cpu = resolv
  kfact_root_cpu = kfact_root
  undermcr_cpu = undermcr
  kk_moy_cpu = kk_moy
  srhs_cpu = srhs
  water2infilt_cpu = water2infilt
  kk_cpu = kk
  stmat_cpu = stmat
  evap_bare_lim_cpu = evap_bare_lim
  evap_bare_lim_ns_cpu = evap_bare_lim_ns
  ksoil_cpu = ksoil
  mc_layh_cpu = mc_layh
  mc_layh_s_cpu = mc_layh_s
  mcl_layh_cpu = mcl_layh
  mcl_layh_s_cpu = mcl_layh_s
  root_deficit_cpu = root_deficit
  us_cpu = us
  CALL declaration_initialization
  CALL read_dummy(altmax, avan, circ_class_biomass, e_frac, evap_bare_lim, evap_bare_lim_ns, evapot, evapot_penm, F_absorption, &
&frac_snow_nobio, humrel, irrigation, ks, mcfc, mcr, mcs, mcw, njsc, nvan, precip_rain, reinf_slope_soil, reinfiltration, &
&returnflow, snow, snowdz, soiltile, stempdiag, tot_bare_soil, tot_melt, totfrac_nobio, tq_cdrag, transpir, u, us, v, veget, &
&veget_max, vevapnu)
  error_flag_hydrol_diag_soil_flux_1 = 0
  error_flag_hydrol_split_soil_1 = 0
  error_flag_hydrol_split_soil_2 = 0
  error_flag_hydrol_split_soil_3 = 0
  error_flag_hydrol_split_soil_4 = 0
  error_flag_hydrol_split_soil_5 = 0
  error_flag_hydrol_split_soil_6 = 0
  error_flag_hydrol_split_soil_7 = 0
  error_flag_hydrol_split_soil_8 = 0
  error_flag_hydrol_split_soil_9 = 0
  error_flag_hydrol_split_soil_10 = 0
  error_flag_hydrol_split_soil_11 = 0
  error_flag_hydrol_split_soil_12 = 0
  error_flag_hydrol_soil_infilt_1 = 0
  error_flag_hydrol_root_profile_1 = 0
  error_flag_hydrol_root_profile_2 = 0
  !$ACC ENTER DATA COPYIN(ks, nvan, avan, mcr, mcs, mcfc, mcw, veget, veget_max, soiltile, njsc, reinf_slope_soil, transpir,  &
!$ACC& vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, reinfiltration, irrigation, tot_melt, evap_bare_lim,  &
!$ACC& evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, stempdiag, snow,  &
!$ACC& snowdz, tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, e_frac, ksoil, altmax, root_profile,  &
!$ACC& root_depth, root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, frac_snow_nobio, F_absorption)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  !$ACC PARALLEL LOOP INDEPENDENT REDUCTION(+:error_flag_hydrol_diag_soil_flux_1, error_flag_hydrol_split_soil_1,  &
!$ACC& error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4,  &
!$ACC& error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7,  &
!$ACC& error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10,  &
!$ACC& error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12, error_flag_hydrol_soil_infilt_1,  &
!$ACC& error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2)
  DO ji = 1, kjpindex
    CALL hydrol_soil_acc(error_flag_hydrol_diag_soil_flux_1, error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, &
&error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, &
&error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, &
&error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12, error_flag_hydrol_soil_infilt_1, &
&error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget, &
&veget_max, soiltile, njsc, reinf_slope_soil, transpir, vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, &
&reinfiltration, irrigation, tot_melt, evap_bare_lim, evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, &
&vegstress, drysoil_frac, stempdiag, snow, snowdz, tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, &
&e_frac, ksoil, altmax, root_profile, root_depth, root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, &
&frac_snow_nobio, F_absorption)
  END DO
  !$ACC END PARALLEL
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  !$ACC UPDATE SELF(mc, dr_ns, mcl, qflux_ns, soil_wet_ns, tmc_litter, vegstressv, soil_wet_litter, tmc, ae_ns, ru_ns, humrelv,  &
!$ACC& profil_froz_hydro_ns, rhs, tmat, resolv, kfact_root, undermcr, kk_moy, srhs, water2infilt, kk, stmat, evap_bare_lim,  &
!$ACC& evap_bare_lim_ns, ksoil, mc_layh, mc_layh_s, mcl_layh, mcl_layh_s, root_deficit, us)
  !$ACC EXIT DATA DELETE(ks, nvan, avan, mcr, mcs, mcfc, mcw, veget, veget_max, soiltile, njsc, reinf_slope_soil, transpir,  &
!$ACC& vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, reinfiltration, irrigation, tot_melt, evap_bare_lim,  &
!$ACC& evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, stempdiag, snow,  &
!$ACC& snowdz, tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, e_frac, ksoil, altmax, root_profile,  &
!$ACC& root_depth, root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, frac_snow_nobio, F_absorption)
  IF (ALL(mc .EQ. mc_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mc_gpu are equal to mc_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mc_gpu do not match mc_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mc - mc_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_gpu:', MINVAL(mc), MAXVAL(mc)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_cpu:', MINVAL(mc_cpu), MAXVAL(mc_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(dr_ns .EQ. dr_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in dr_ns_gpu are equal to dr_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in dr_ns_gpu do not match dr_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(dr_ns - dr_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of dr_ns_gpu:', MINVAL(dr_ns), MAXVAL(dr_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of dr_ns_cpu:', MINVAL(dr_ns_cpu), MAXVAL(dr_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mcl .EQ. mcl_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mcl_gpu are equal to mcl_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mcl_gpu do not match mcl_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mcl - mcl_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_gpu:', MINVAL(mcl), MAXVAL(mcl)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_cpu:', MINVAL(mcl_cpu), MAXVAL(mcl_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(qflux_ns .EQ. qflux_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in qflux_ns_gpu are equal to qflux_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in qflux_ns_gpu do not match qflux_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(qflux_ns - qflux_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qflux_ns_gpu:', MINVAL(qflux_ns), MAXVAL(qflux_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qflux_ns_cpu:', MINVAL(qflux_ns_cpu), MAXVAL(qflux_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(soil_wet_ns .EQ. soil_wet_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in soil_wet_ns_gpu are equal to soil_wet_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in soil_wet_ns_gpu do not match soil_wet_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(soil_wet_ns - soil_wet_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soil_wet_ns_gpu:', MINVAL(soil_wet_ns), MAXVAL(soil_wet_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soil_wet_ns_cpu:', MINVAL(soil_wet_ns_cpu), MAXVAL(soil_wet_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmc_litter .EQ. tmc_litter_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_litter_gpu are equal to tmc_litter_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_litter_gpu do not match tmc_litter_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc_litter - tmc_litter_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litter_gpu:', MINVAL(tmc_litter), MAXVAL(tmc_litter)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litter_cpu:', MINVAL(tmc_litter_cpu), MAXVAL(tmc_litter_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(vegstressv .EQ. vegstressv_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in vegstressv_gpu are equal to vegstressv_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in vegstressv_gpu do not match vegstressv_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(vegstressv - vegstressv_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vegstressv_gpu:', MINVAL(vegstressv), MAXVAL(vegstressv)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vegstressv_cpu:', MINVAL(vegstressv_cpu), MAXVAL(vegstressv_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(soil_wet_litter .EQ. soil_wet_litter_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in soil_wet_litter_gpu are equal to soil_wet_litter_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in soil_wet_litter_gpu do not match soil_wet_litter_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(soil_wet_litter - soil_wet_litter_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soil_wet_litter_gpu:', MINVAL(soil_wet_litter), MAXVAL(soil_wet_litter)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soil_wet_litter_cpu:', MINVAL(soil_wet_litter_cpu), MAXVAL(soil_wet_litter_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmc .EQ. tmc_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_gpu are equal to tmc_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_gpu do not match tmc_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc - tmc_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_gpu:', MINVAL(tmc), MAXVAL(tmc)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_cpu:', MINVAL(tmc_cpu), MAXVAL(tmc_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(ae_ns .EQ. ae_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ae_ns_gpu are equal to ae_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ae_ns_gpu do not match ae_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ae_ns - ae_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ae_ns_gpu:', MINVAL(ae_ns), MAXVAL(ae_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ae_ns_cpu:', MINVAL(ae_ns_cpu), MAXVAL(ae_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(ru_ns .EQ. ru_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ru_ns_gpu are equal to ru_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ru_ns_gpu do not match ru_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ru_ns - ru_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ru_ns_gpu:', MINVAL(ru_ns), MAXVAL(ru_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ru_ns_cpu:', MINVAL(ru_ns_cpu), MAXVAL(ru_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(humrelv .EQ. humrelv_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in humrelv_gpu are equal to humrelv_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in humrelv_gpu do not match humrelv_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(humrelv - humrelv_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humrelv_gpu:', MINVAL(humrelv), MAXVAL(humrelv)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humrelv_cpu:', MINVAL(humrelv_cpu), MAXVAL(humrelv_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(profil_froz_hydro_ns .EQ. profil_froz_hydro_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in profil_froz_hydro_ns_gpu are equal to profil_froz_hydro_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in profil_froz_hydro_ns_gpu do not match profil_froz_hydro_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(profil_froz_hydro_ns - profil_froz_hydro_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_ns_gpu:', MINVAL(profil_froz_hydro_ns), MAXVAL(profil_froz_hydro_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_ns_cpu:', MINVAL(profil_froz_hydro_ns_cpu), &
&MAXVAL(profil_froz_hydro_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(rhs .EQ. rhs_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in rhs_gpu are equal to rhs_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in rhs_gpu do not match rhs_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(rhs - rhs_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rhs_gpu:', MINVAL(rhs), MAXVAL(rhs)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rhs_cpu:', MINVAL(rhs_cpu), MAXVAL(rhs_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmat .EQ. tmat_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmat_gpu are equal to tmat_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmat_gpu do not match tmat_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmat - tmat_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmat_gpu:', MINVAL(tmat), MAXVAL(tmat)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmat_cpu:', MINVAL(tmat_cpu), MAXVAL(tmat_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(resolv .EQV. resolv_cpu)) THEN
    WRITE(*, *) 'LOGICAL EQV test passed: All elements in resolv_gpu are equal to resolv_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'LOGICAL EQV test failed: Not all elements in resolv_gpu match resolv_cpu.'
    WRITE(*, *) ''
  END IF
  IF (ALL(kfact_root .EQ. kfact_root_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in kfact_root_gpu are equal to kfact_root_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in kfact_root_gpu do not match kfact_root_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(kfact_root - kfact_root_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kfact_root_gpu:', MINVAL(kfact_root), MAXVAL(kfact_root)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kfact_root_cpu:', MINVAL(kfact_root_cpu), MAXVAL(kfact_root_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(undermcr .EQ. undermcr_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in undermcr_gpu are equal to undermcr_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in undermcr_gpu do not match undermcr_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(undermcr - undermcr_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of undermcr_gpu:', MINVAL(undermcr), MAXVAL(undermcr)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of undermcr_cpu:', MINVAL(undermcr_cpu), MAXVAL(undermcr_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(kk_moy .EQ. kk_moy_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in kk_moy_gpu are equal to kk_moy_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in kk_moy_gpu do not match kk_moy_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(kk_moy - kk_moy_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kk_moy_gpu:', MINVAL(kk_moy), MAXVAL(kk_moy)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kk_moy_cpu:', MINVAL(kk_moy_cpu), MAXVAL(kk_moy_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(srhs .EQ. srhs_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in srhs_gpu are equal to srhs_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in srhs_gpu do not match srhs_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(srhs - srhs_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of srhs_gpu:', MINVAL(srhs), MAXVAL(srhs)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of srhs_cpu:', MINVAL(srhs_cpu), MAXVAL(srhs_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(water2infilt .EQ. water2infilt_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in water2infilt_gpu are equal to water2infilt_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in water2infilt_gpu do not match water2infilt_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(water2infilt - water2infilt_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of water2infilt_gpu:', MINVAL(water2infilt), MAXVAL(water2infilt)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of water2infilt_cpu:', MINVAL(water2infilt_cpu), MAXVAL(water2infilt_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(kk .EQ. kk_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in kk_gpu are equal to kk_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in kk_gpu do not match kk_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(kk - kk_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kk_gpu:', MINVAL(kk), MAXVAL(kk)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of kk_cpu:', MINVAL(kk_cpu), MAXVAL(kk_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(stmat .EQ. stmat_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in stmat_gpu are equal to stmat_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in stmat_gpu do not match stmat_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(stmat - stmat_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of stmat_gpu:', MINVAL(stmat), MAXVAL(stmat)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of stmat_cpu:', MINVAL(stmat_cpu), MAXVAL(stmat_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(evap_bare_lim .EQ. evap_bare_lim_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in evap_bare_lim_gpu are equal to evap_bare_lim_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in evap_bare_lim_gpu do not match evap_bare_lim_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(evap_bare_lim - evap_bare_lim_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of evap_bare_lim_gpu:', MINVAL(evap_bare_lim), MAXVAL(evap_bare_lim)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of evap_bare_lim_cpu:', MINVAL(evap_bare_lim_cpu), MAXVAL(evap_bare_lim_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(evap_bare_lim_ns .EQ. evap_bare_lim_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in evap_bare_lim_ns_gpu are equal to evap_bare_lim_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in evap_bare_lim_ns_gpu do not match evap_bare_lim_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(evap_bare_lim_ns - evap_bare_lim_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of evap_bare_lim_ns_gpu:', MINVAL(evap_bare_lim_ns), MAXVAL(evap_bare_lim_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of evap_bare_lim_ns_cpu:', MINVAL(evap_bare_lim_ns_cpu), MAXVAL(evap_bare_lim_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(ksoil .EQ. ksoil_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ksoil_gpu are equal to ksoil_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ksoil_gpu do not match ksoil_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ksoil - ksoil_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ksoil_gpu:', MINVAL(ksoil), MAXVAL(ksoil)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ksoil_cpu:', MINVAL(ksoil_cpu), MAXVAL(ksoil_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mc_layh .EQ. mc_layh_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mc_layh_gpu are equal to mc_layh_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mc_layh_gpu do not match mc_layh_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mc_layh - mc_layh_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_layh_gpu:', MINVAL(mc_layh), MAXVAL(mc_layh)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_layh_cpu:', MINVAL(mc_layh_cpu), MAXVAL(mc_layh_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mc_layh_s .EQ. mc_layh_s_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mc_layh_s_gpu are equal to mc_layh_s_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mc_layh_s_gpu do not match mc_layh_s_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mc_layh_s - mc_layh_s_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_layh_s_gpu:', MINVAL(mc_layh_s), MAXVAL(mc_layh_s)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_layh_s_cpu:', MINVAL(mc_layh_s_cpu), MAXVAL(mc_layh_s_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mcl_layh .EQ. mcl_layh_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mcl_layh_gpu are equal to mcl_layh_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mcl_layh_gpu do not match mcl_layh_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mcl_layh - mcl_layh_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_layh_gpu:', MINVAL(mcl_layh), MAXVAL(mcl_layh)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_layh_cpu:', MINVAL(mcl_layh_cpu), MAXVAL(mcl_layh_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mcl_layh_s .EQ. mcl_layh_s_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mcl_layh_s_gpu are equal to mcl_layh_s_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mcl_layh_s_gpu do not match mcl_layh_s_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mcl_layh_s - mcl_layh_s_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_layh_s_gpu:', MINVAL(mcl_layh_s), MAXVAL(mcl_layh_s)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_layh_s_cpu:', MINVAL(mcl_layh_s_cpu), MAXVAL(mcl_layh_s_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(root_deficit .EQ. root_deficit_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in root_deficit_gpu are equal to root_deficit_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in root_deficit_gpu do not match root_deficit_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(root_deficit - root_deficit_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of root_deficit_gpu:', MINVAL(root_deficit), MAXVAL(root_deficit)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of root_deficit_cpu:', MINVAL(root_deficit_cpu), MAXVAL(root_deficit_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(us .EQ. us_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in us_gpu are equal to us_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in us_gpu do not match us_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(us - us_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of us_gpu:', MINVAL(us), MAXVAL(us)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of us_cpu:', MINVAL(us_cpu), MAXVAL(us_cpu)
    WRITE(*, *) ''
  END IF
  IF (error_flag_hydrol_diag_soil_flux_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_diag_soil_flux, error_flag_hydrol_diag_soil_flux_1 is > 0 :', &
&error_flag_hydrol_diag_soil_flux_1
    CALL ipslerr_p(1, 'hydrol_diag_soil_flux', 'NOTE:', 'Problem in the water balance, qflux_ns computation', '')
  END IF
  IF (error_flag_hydrol_split_soil_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_1 is > 0 :', error_flag_hydrol_split_soil_1
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'PRECISOL SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_2 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_2 is > 0 :', error_flag_hydrol_split_soil_2
  END IF
  IF (error_flag_hydrol_split_soil_3 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_3 is > 0 :', error_flag_hydrol_split_soil_3
  END IF
  IF (error_flag_hydrol_split_soil_4 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_4 is > 0 :', error_flag_hydrol_split_soil_4
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'VEVAPNU SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_5 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_5 is > 0 :', error_flag_hydrol_split_soil_5
  END IF
  IF (error_flag_hydrol_split_soil_6 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_6 is > 0 :', error_flag_hydrol_split_soil_6
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'TRANSPIR SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_7 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_7 is > 0 :', error_flag_hydrol_split_soil_7
  END IF
  IF (error_flag_hydrol_split_soil_8 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_8 is > 0 :', error_flag_hydrol_split_soil_8
  END IF
  IF (error_flag_hydrol_split_soil_9 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_9 is > 0 :', error_flag_hydrol_split_soil_9
  END IF
  IF (error_flag_hydrol_split_soil_10 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_10 is > 0 :', error_flag_hydrol_split_soil_10
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'ROOTSINK SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_11 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_11 is > 0 :', error_flag_hydrol_split_soil_11
  END IF
  IF (error_flag_hydrol_split_soil_12 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_12 is > 0 :', error_flag_hydrol_split_soil_12
    CALL ipslerr_p(3, 'hydrol_split_soil', 'We will STOP now.', 'One or several fatal errors were found previously.', '')
  END IF
  IF (error_flag_hydrol_soil_infilt_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_soil_infilt, error_flag_hydrol_soil_infilt_1 is > 0 :', error_flag_hydrol_soil_infilt_1
    CALL ipslerr_p(3, 'hydrol_soil_infilt', 'We will STOP now.', 'Error in calculation of infilt tot', '')
  END IF
  IF (error_flag_hydrol_root_profile_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_root_profile, error_flag_hydrol_root_profile_1 is > 0 :', &
&error_flag_hydrol_root_profile_1
    CALL ipslerr_p(plev, 'hydrol.f90', 'structural root profile does not add up to 1', 'Check its calculation', '')
  END IF
  IF (error_flag_hydrol_root_profile_2 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_root_profile, error_flag_hydrol_root_profile_2 is > 0 :', &
&error_flag_hydrol_root_profile_2
    CALL ipslerr_p(plev, 'hydrol.f90', 'functional root profile does not add up to 1', 'Check its calculation', '')
  END IF
  CONTAINS


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_setup
  !!
  !>\BRIEF        This subroutine computes the matrix coef.
  !!
  !! DESCRIPTION  : None
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S) : matrix coef
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================

  SUBROUTINE hydrol_soil_setup_acc(ji, kjpindex, ins)
    !$ACC ROUTINE SEQ


    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: temp4
    REAL(KIND = r_std) :: temp3

    !_
    !& 
!& ================================================================================================================================
    !-we compute tridiag matrix coefficients (LEFT and RIGHT)
    ! of the system to solve [LEFT]*mc_{t+1}=[RIGHT]*mc{t}+[add terms]:
    ! e(nslm),f(nslm),g1(nslm) for the [left] vector
    ! and ep(nslm),fp(nslm),gp(nslm) for the [right] vector

    ! w_time=1 (in constantes_soil) indicates implicit computation for diffusion
    temp3 = w_time * (dt_sechiba / one_day) / deux
    temp4 = (un - w_time) * (dt_sechiba / one_day) / deux

    ! Passage to arithmetic means for layer averages also in this subroutine : Aurelien 11/05/10

    !- coefficient for first layer
    e(ji, 1) = zero
    f(ji, 1) = trois * dz(2) / huit + temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
    g1(ji, 1) = dz(2) / (huit) - temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
    ep(ji, 1) = zero
    fp(ji, 1) = trois * dz(2) / huit - temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
    gp(ji, 1) = dz(2) / (huit) + temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))

      !- coefficient for medium layers

      DO jsl = 2, nslm - 1
      e(ji, jsl) = dz(jsl) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

      f(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit + temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + &
&d(ji, jsl + 1)) / (dz(jsl + 1)))

      g1(ji, jsl) = dz(jsl + 1) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))

      ep(ji, jsl) = dz(jsl) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

      fp(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit - temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + &
&d(ji, jsl + 1)) / (dz(jsl + 1)))

      gp(ji, jsl) = dz(jsl + 1) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))
    END DO

    !- coefficient for last layer
    e(ji, nslm) = dz(nslm) / (huit) - temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
    f(ji, nslm) = trois * dz(nslm) / huit + temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * &
&free_drain_coef(ji, ins)))
    g1(ji, nslm) = zero
    ep(ji, nslm) = dz(nslm) / (huit) + temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
    fp(ji, nslm) = trois * dz(nslm) / huit - temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * &
&free_drain_coef(ji, ins)))
    gp(ji, nslm) = zero

  END SUBROUTINE hydrol_soil_setup_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_setup
    !!
    !>\BRIEF        This subroutine computes the matrix coef.
    !!
    !! DESCRIPTION  : None
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S) : matrix coef
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================

    SUBROUTINE hydrol_soil_setup(kjpindex, ins)


    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: temp4
    REAL(KIND = r_std) :: temp3

    !_
    !& 
!& ================================================================================================================================
    !-we compute tridiag matrix coefficients (LEFT and RIGHT)
    ! of the system to solve [LEFT]*mc_{t+1}=[RIGHT]*mc{t}+[add terms]:
    ! e(nslm),f(nslm),g1(nslm) for the [left] vector
    ! and ep(nslm),fp(nslm),gp(nslm) for the [right] vector

    ! w_time=1 (in constantes_soil) indicates implicit computation for diffusion
    temp3 = w_time * (dt_sechiba / one_day) / deux
    temp4 = (un - w_time) * (dt_sechiba / one_day) / deux

      ! Passage to arithmetic means for layer averages also in this subroutine : Aurelien 11/05/10

      !- coefficient for first layer
      DO ji = 1, kjpindex
      e(ji, 1) = zero
      f(ji, 1) = trois * dz(2) / huit + temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
      g1(ji, 1) = dz(2) / (huit) - temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
      ep(ji, 1) = zero
      fp(ji, 1) = trois * dz(2) / huit - temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
      gp(ji, 1) = dz(2) / (huit) + temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
    END DO

      !- coefficient for medium layers

      DO jsl = 2, nslm - 1
      DO ji = 1, kjpindex
        e(ji, jsl) = dz(jsl) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

        f(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit + temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + &
&d(ji, jsl + 1)) / (dz(jsl + 1)))

        g1(ji, jsl) = dz(jsl + 1) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))

        ep(ji, jsl) = dz(jsl) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

        fp(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit - temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + &
&d(ji, jsl + 1)) / (dz(jsl + 1)))

        gp(ji, jsl) = dz(jsl + 1) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))
      END DO
    END DO

      !- coefficient for last layer
      DO ji = 1, kjpindex
      e(ji, nslm) = dz(nslm) / (huit) - temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
      f(ji, nslm) = trois * dz(nslm) / huit + temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * &
&free_drain_coef(ji, ins)))
      g1(ji, nslm) = zero
      ep(ji, nslm) = dz(nslm) / (huit) + temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
      fp(ji, nslm) = trois * dz(nslm) / huit - temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * &
&free_drain_coef(ji, ins)))
      gp(ji, nslm) = zero
    END DO

  END SUBROUTINE hydrol_soil_setup



  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_root_profile
  !!
  !>\BRIEF         Calculates the share of the root biomass in each soil layer based on
  !!               structural and functional approach.Calculate the root profile
  !!
  !! DESCRIPTION  : Root structure is probably how most
  !! of us think about roots (i.e. digging a whole and observing where the roots
  !! are). When thinking at root structure, the profile should be relatively
  !! constant over time. A logic time integrator to determine this constancy is
  !! longevity_root as the profile cannot grow faster then the roots grow and die.
  !! Currently the structural root profile is simply fixed it over time. This could
  !! be changed by, for example, making humcste a function of the tree diameter.

  !! In ORCHIDEE root structure is used in the calculation of kfact_root which is water
  !! infiltration along roots (accounted for in hydrol.f90) and the input of soil
  !! carbon and nitrogen at depth due to the turnover of roots which is accounted
  !! for stomate_soil_carbon_discretization.f90. Furthermore, it is used to calculate
  !! the root temperature in stomate_resp.f90. When thinking about root function it
  !! is not so important where the roots are located but it is more important at
  !! which depth the roots will be active. The function approach could be used to
  !! calculate from which soil layers the plants take most the soil water for their
  !! transpiration. This way of looking at the roots is similar to how we look at
  !! the canopy where we have a lot of leaves at places in the canopy where little
  !! light can penetrate and where a large part of the photosynthesis is taken care
  !! of by the leaves in the top layers of the canopy.
  !!
  !! NOTE: for the moment root structure and root function are only coupled through
  !! the depth of the soil. In the absence of roots, the root functions cannot be
  !! fullfilled. This is the most minimalistic coupling. It basically implies that
  !! a very small fraction of the roots, e.g., < 1% could take up all the water
  !! required for transpiration. A thighter coupling between structure and function
  !! is to be expected but this needs to be checked in the literature.
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S) : root_profile
  !!
  !! REFERENCE(S) :
  !!
  !! \n
  !_
  !& 
!& ================================================================================================================================

  SUBROUTINE hydrol_root_profile_acc(error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2, ji, kjpindex, altmax, sm, &
&smw, root_profile, root_depth)
    !$ACC ROUTINE SEQ

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_2
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: altmax
    !! Maximul active layer thickness (m). Be careful, here active means not frozen.
    !! Not related with the active soil carbon pool.
    REAL(KIND = r_std), DIMENSION(nslm), INTENT(IN) :: sm
    !! Soil moisture of each layer (liquid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm), INTENT(IN) :: smw
    !! Soil moisture of each layer at wilting point
    !!  @tex $(kg m^{-2})$ @endtex

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof), INTENT(OUT) :: root_profile
    !! Normalized root mass/length fraction in each soil layer
    !! (0-1, unitless)
    !! DIM = kjpindex * nvm * nslm
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ndepths), INTENT(OUT) :: root_depth
    !! Node and interface numbers at which the deepest roots
    !! occur (1 to nslm, unitless)


    !! 0.3 Modified variables

    !! 0.4 Local variables
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    !! Indices
    REAL(KIND = r_std) :: rpc
    !! Integration constant for vertical decomposer
    REAL(KIND = r_std) :: z_bottom
    REAL(KIND = r_std) :: z_top
    !! top and bottom node in between which to integrate the root profile
    REAL(KIND = r_std) :: count
    !! Count the number of errors
    REAL(KIND = r_std), DIMENSION(nslm) :: root_profile_tmp
    !! Temporary variable
    REAL(KIND = r_std) :: root_depth_tmp
    REAL(KIND = r_std) :: minmax_value
    INTEGER(KIND = i_std) :: minmax_index
    !! Temporary variable

    !_
    !& 
!& ================================================================================================================================

    !! 1.Rooting depth

    ! Calculate the maximum depth roots occur at. Two constraints are already accouned
    ! for: (1) crop can root to 0.8 m, grasslands are assumed to root no deeper than 1 m.
    ! Trees can root down to 2 m. (2) Roots do not extend into frozen soils.
    ! ORCHIDEE assumes that plant roots go down to the depth of the soil water profile
    ! right from day 1. In other words, even a very young tree sapling, crop or grasslands
    ! has roots that extend down to their :: max_root_depth.
    ! Hence, plant age/height does NOT constraint rooting depth for now. A future
    ! development could make rooting depth a function of plant height. This approach
    ! would correctly increase the water stress of young vegetation but may wrongly
    ! increase the water stress of young plants growing in semi-arid regions. In such
    ! regions vegetative reproduction is likley to be common as it allows the offspring
    ! to use water from the parent plant as long as the offsprong is too small to reach
    ! the deeper water layers.
    root_depth(ji, :, :) = zero
    DO jv = 1, nvm

      ! Plants can root up to 2 m (zdr(jsl), the prescribed max_root_depth
      ! or the first permafrost layer
      root_depth_tmp = MIN(MIN(altmax(ji, jv), maxaltmax), MIN(zdr(nslm), max_root_depth(jv)))

      ! Find the index of the node which is the closest to the actual
      ! root_depth. This layer is used to truncate the root profile.
      minmax_value = HUGE(0.0)
      minmax_index = 1
      DO jsl = 1, nslm
        IF (ABS(root_depth_tmp - znh(jsl)) .LT. minmax_value) THEN
          minmax_value = ABS(root_depth_tmp - znh(jsl))
          minmax_index = jsl + 1 - 1
        END IF
      END DO
      root_depth(ji, jv, inode) = minmax_index

      ! zdr is defined as a 0:nslm matrix. MINLOC does not know that
      ! and gives the indices ad 1:nslm+1. Convert the indices back to
      ! 0:nslm by subtracting 1.
      minmax_value = HUGE(0.0)
      minmax_index = 1
      DO jsl = 0, nslm
        IF (ABS(root_depth_tmp - zdr(jsl)) .LT. minmax_value) THEN
          minmax_value = ABS(root_depth_tmp - zdr(jsl))
          minmax_index = jsl + 1 - 0
        END IF
      END DO
      root_depth(ji, jv, iinterface) = minmax_index - 1

    END DO
    ! jv
      DO jv = 1, nvm
      IF (root_depth(ji, jv, inode) .LE. 2) THEN
        root_depth(ji, jv, inode) = 2
      END IF
    END DO
    DO jv = 1, nvm
      IF (root_depth(ji, jv, iinterface) .LE. 2) THEN
        root_depth(ji, jv, iinterface) = 2
      END IF
    END DO


    !! 2.Structural root profile

    ! The structural root profile is calculated as an exponentially decreasing
    ! root mass with depth which. ORCHIDEE uses the structural root profile to
    ! calculate rain infiltration along the roots, som inputs at depth and the
    ! root temperature for autotrophic respiration.

    ! NOTE: the shape of the profile is determined by its depth and the parameter
    ! humcste. For the moment humcste is PFT-dependent but constant over time. The
    ! depth is PFT-dependent (see above) and is a function of the active layer
    ! thickness when the soil discretization is used. The depth of the root profile
    ! could/should be made a function of tree diameter or plant biomass.
    root_profile(ji, :, :, istruc) = zero
    DO jv = 1, nvm

      ! Note that the integration will start at the top of the second layer (the
      ! top of the first layer is zdr(0) hence the zdr(1) is the top of the
      ! second layer) and will continue until the bottom of the profile. The
      ! bottom is the profile is calculated above but never extends deeper than
      ! the depth used in hydrol.f90 (in thermosoil.f90 the profile extends much
      ! deeper. The first layer is excluded because the profile will be used to
      ! extract water from the soil. The first layer is very thin and by extracting
      ! water it could dry to quickly. Also in reality not too many roots are found
      ! in the top mm of the soil. zdr describes the nodes and interfaces of the soil
      ! layers as proposed by de Rosnay 1999 (PhD thesis).
      z_top = zdr(1)
      z_bottom = zdr(root_depth(ji, jv, iinterface))

      ! Calculate the total surface area under an exponential curve between
      ! zdr(1) and the zdr(nslm)
      rpc = un / (EXP(- z_top * humcste(jv)) - EXP(- z_bottom * humcste(jv)))

        DO jsl = 2, root_depth(ji, jv, iinterface)

        ! Calculate the share of the total roots for layers which are "centered"
        ! at the nodes of the hydrology scheme. Centered was written in quotes
        ! because the layer is not symmetric. Using the nodes as the center of the
        ! layers follows De Rosnay (PhD, figure C.2 page 156). The root profile
        ! starts at the top of the second layer, ends at the bottom of the 11th
        ! layer and is calculated for the nodes in between.
        ! The following equation are derived (but rewritten) from the integrals
        ! of equations C9 to C11 of De Rosnay's (1999) PhD thesis (page 158).
        root_profile(ji, jv, jsl, istruc) = rpc * (EXP(- zdr(jsl - 1) * humcste(jv)) - EXP(- zdr(jsl) * humcste(jv)))

      END DO
      ! root_depth

      ! Top layer does not contain structural roots (see z_top)
      ! This line is not needed but was added as a reminder.
      root_profile(ji, jv, 1, istruc) = zero

        ! Error checking. Each root profile should add up to 1.
        IF (err_act .GT. 1) THEN
        IF (ABS(SUM(root_profile(ji, jv, :, istruc)) - un) .GT. 100 * EPSILON(un)) THEN
          error_flag_hydrol_root_profile_1 = error_flag_hydrol_root_profile_1 + 1
        END IF
      END IF

      ! kjpindex
    END DO
    ! nvm


    !! 3. Functional root profile

    ! Calculates the share of the root biomass in each soil layer based on
    ! the soil water content in each layer. The roots now follow the water.
    ! this results in a very dynamic root profile that may change every half
    ! hour (i.e. the time step for hydrol). ORCHIDEE uses the root function
    ! to calculate plant water stress (hydraulic_arch.f90) and the soil layers
    ! from which the transpiration is taken (hydrol.f90).

    ! NOTE: yet anonther root function is nutrient uptake. A separate root
    ! profile could be calculated to be used in the calculation of N uptake
    ! is stomate_soilcarbon.f90 (nitrogen_dynamics).
    root_profile(ji, :, :, ifunc) = zero
    DO jv = 2, nvm

      ! Plant available soil water per layer.
      ! The calculations could make use of sm and smw. These variables has
      ! been labelled soil moisture in kg/m2 and soil moisture at each layer
      ! at wilting point also in kg/m2. As an alternative swc and mcr could
      ! be used. swc is calculated in hydrol.f90 based on mc (m3/m3). mc is
      ! used to calculate smt (total soil water thus liquid + ice) and mcl is
      ! used to calculate sm (liquid only). sm denotes only liquid water, swc
      ! denotes liquid and frozen water. Given the focus on root function, a
      ! variable describing the liquid water seems the better choice.

      !+++CHECK+++
      ! The most important seems to be the difference in the dimensions of the
      ! variables: the dimensions of sm variables are kjpindex,nslm the
      ! dimension of the swc variable is kjpindex,nslm,nst. For the application
      ! we have in mind a different root profile for each soil tile (nst) seems
      ! desirable. Note that the difference in dimensions reflect the spatial
      ! scale of the model but it is not clear at which scale one approach
      ! (tile vs pixel) would be really prefered above another. Should we use
      ! the commented code further below?
      root_profile_tmp(:) = zero
      DO jsl = 2, root_depth(ji, jv, iinterface)
        root_profile_tmp(jsl) = MAX(zero, sm(jsl) - smw(jsl))
      END DO
      !+++++++++++

        ! Normalize to obtain a root profile (fraction between 0 and 1)
        IF (SUM(root_profile_tmp(:)) .GT. min_sechiba) THEN
        root_profile(ji, jv, :, ifunc) = root_profile_tmp(:) / SUM(root_profile_tmp(:))
      ELSE
        root_profile(ji, jv, :, ifunc) = zero
      END IF

      ! Top layer is not used for water uptake
      ! This line is not needed but was added as a reminder.
      root_profile(ji, jv, 1, ifunc) = zero

        ! The functional profile should also equal to one if there is soil moisture.
        IF (err_act .GT. 1) THEN
        IF (ABS(SUM(root_profile(ji, jv, :, ifunc)) - un) .GT. 100 * EPSILON(un) .AND. SUM(root_profile_tmp(:)) .GT. min_sechiba) &
&THEN
          error_flag_hydrol_root_profile_2 = error_flag_hydrol_root_profile_2 + 1
        END IF
      END IF

      ! kjpindex
    END DO
    ! ivm


    !+++CHECK+++
    ! The code above gives one root profile per pixel. The approach below would
    ! enable calculating one root profile per soil tile (bare, short and
    ! tall vegetation). NOTE that the variable name needs to be changed
    ! from root_dens to root_profile and that the dimensions need to re-
    ! ordered.
    !!$    DO ivm = 1, nvm
    !!$
    !!$       ! Link the pft to the soil tile
    !!$       istm = pref_soil_veg(ivm)
    !!$       IF ( is_tree(ivm) ) THEN
    !!$
    !!$          nroot_tmp(:) = zero
    !!$          ! Plant available soil water per layer. mcr is the residual soil
    !!$          ! water and depends on the soil type which is stored in njsc(ipts)
    !!$          DO ibdl = 2, nslm
    !!$             nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
    !!$          ENDDO
    !!$
    !!$       ELSE
    !!$
    !!$          ! Specific case for grasses where we only consider the first 1m of soil.
    !!$          ! Plant available soil water per layer. mcr is the residual soil
    !!$          ! water and depends on the soil type which is stored in njsc(ipts)
    !!$          nroot_tmp(:) = zero
    !!$          DO ibdl = 2, nslm
    !!$             IF (znt(ibdl) .LT. un) THEN
    !!$                nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
    !!$             ELSE
    !!$                nroot_tmp(ibdl) = zero
    !!$             END IF
    !!$          ENDDO
    !!$
    !!$       END IF
    !!$
    !!$       ! Normalize to obtain a root profile (fraction between 0 and 1)
    !!$       IF (SUM(nroot_tmp(:)) .GT. min_sechiba ) THEN
    !!$          root_dens(ipts,:,ivm) = nroot_tmp(:)/SUM(nroot_tmp(:))
    !!$       ELSE
    !!$          root_dens(ipts,:,ivm) = zero
    !!$       END IF
    !!$       root_dens(ipts,1,ivm) = zero
    !!$
    !!$    ENDDO
    !+++++++++++


  END SUBROUTINE hydrol_root_profile_acc



    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_root_profile
    !!
    !>\BRIEF         Calculates the share of the root biomass in each soil layer based on
    !!               structural and functional approach.Calculate the root profile
    !!
    !! DESCRIPTION  : Root structure is probably how most
    !! of us think about roots (i.e. digging a whole and observing where the roots
    !! are). When thinking at root structure, the profile should be relatively
    !! constant over time. A logic time integrator to determine this constancy is
    !! longevity_root as the profile cannot grow faster then the roots grow and die.
    !! Currently the structural root profile is simply fixed it over time. This could
    !! be changed by, for example, making humcste a function of the tree diameter.

    !! In ORCHIDEE root structure is used in the calculation of kfact_root which is water
    !! infiltration along roots (accounted for in hydrol.f90) and the input of soil
    !! carbon and nitrogen at depth due to the turnover of roots which is accounted
    !! for stomate_soil_carbon_discretization.f90. Furthermore, it is used to calculate
    !! the root temperature in stomate_resp.f90. When thinking about root function it
    !! is not so important where the roots are located but it is more important at
    !! which depth the roots will be active. The function approach could be used to
    !! calculate from which soil layers the plants take most the soil water for their
    !! transpiration. This way of looking at the roots is similar to how we look at
    !! the canopy where we have a lot of leaves at places in the canopy where little
    !! light can penetrate and where a large part of the photosynthesis is taken care
    !! of by the leaves in the top layers of the canopy.
    !!
    !! NOTE: for the moment root structure and root function are only coupled through
    !! the depth of the soil. In the absence of roots, the root functions cannot be
    !! fullfilled. This is the most minimalistic coupling. It basically implies that
    !! a very small fraction of the roots, e.g., < 1% could take up all the water
    !! required for transpiration. A thighter coupling between structure and function
    !! is to be expected but this needs to be checked in the literature.
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S) : root_profile
    !!
    !! REFERENCE(S) :
    !!
    !! \n
    !_
    !& 
!& ================================================================================================================================

    SUBROUTINE hydrol_root_profile(kjpindex, altmax, sm, smw, root_profile, root_depth)

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    REAL(KIND = r_std), DIMENSION(:, :), INTENT(IN) :: altmax
    !! Maximul active layer thickness (m). Be careful, here active means not frozen.
    !! Not related with the active soil carbon pool.
    REAL(KIND = r_std), DIMENSION(:, :), INTENT(IN) :: sm
    !! Soil moisture of each layer (liquid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(:, :), INTENT(IN) :: smw
    !! Soil moisture of each layer at wilting point
    !!  @tex $(kg m^{-2})$ @endtex

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(:, :, :, :), INTENT(OUT) :: root_profile
    !! Normalized root mass/length fraction in each soil layer
    !! (0-1, unitless)
    !! DIM = kjpindex * nvm * nslm
    REAL(KIND = r_std), DIMENSION(:, :, :), INTENT(OUT) :: root_depth
    !! Node and interface numbers at which the deepest roots
    !! occur (1 to nslm, unitless)


    !! 0.3 Modified variables

    !! 0.4 Local variables
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: ji
    !! Indices
    REAL(KIND = r_std) :: rpc
    !! Integration constant for vertical decomposer
    REAL(KIND = r_std) :: z_bottom
    REAL(KIND = r_std) :: z_top
    !! top and bottom node in between which to integrate the root profile
    REAL(KIND = r_std), DIMENSION(kjpindex) :: count
    !! Count the number of errors
    REAL(KIND = r_std), DIMENSION(nslm) :: root_profile_tmp
    !! Temporary variable
    REAL(KIND = r_std) :: root_depth_tmp
    !! Temporary variable

    !_
    !& 
!& ================================================================================================================================

    !! 1.Rooting depth

    ! Calculate the maximum depth roots occur at. Two constraints are already accouned
    ! for: (1) crop can root to 0.8 m, grasslands are assumed to root no deeper than 1 m.
    ! Trees can root down to 2 m. (2) Roots do not extend into frozen soils.
    ! ORCHIDEE assumes that plant roots go down to the depth of the soil water profile
    ! right from day 1. In other words, even a very young tree sapling, crop or grasslands
    ! has roots that extend down to their :: max_root_depth.
    ! Hence, plant age/height does NOT constraint rooting depth for now. A future
    ! development could make rooting depth a function of plant height. This approach
    ! would correctly increase the water stress of young vegetation but may wrongly
    ! increase the water stress of young plants growing in semi-arid regions. In such
    ! regions vegetative reproduction is likley to be common as it allows the offspring
    ! to use water from the parent plant as long as the offsprong is too small to reach
    ! the deeper water layers.
    root_depth(:, :, :) = zero
    DO ji = 1, kjpindex
      DO jv = 1, nvm

        ! Plants can root up to 2 m (zdr(jsl), the prescribed max_root_depth
        ! or the first permafrost layer
        root_depth_tmp = MIN(MIN(altmax(ji, jv), maxaltmax), MIN(zdr(nslm), max_root_depth(jv)))

        ! Find the index of the node which is the closest to the actual
        ! root_depth. This layer is used to truncate the root profile.
        root_depth(ji, jv, inode) = MINLOC(ABS(root_depth_tmp - znh(:)), DIM = 1)

        ! zdr is defined as a 0:nslm matrix. MINLOC does not know that
        ! and gives the indices ad 1:nslm+1. Convert the indices back to
        ! 0:nslm by subtracting 1.
        root_depth(ji, jv, iinterface) = MINLOC(ABS(root_depth_tmp - zdr(:)), DIM = 1) - 1

      END DO
      ! jv
    END DO
    ! ji

      ! Prescribe a solution for very shallow root systems
      WHERE (root_depth(:, :, inode) .LE. 2)
      root_depth(:, :, inode) = 2
    END WHERE

      WHERE (root_depth(:, :, iinterface) .LE. 2)
      root_depth(:, :, iinterface) = 2
    END WHERE


    !! 2.Structural root profile

    ! The structural root profile is calculated as an exponentially decreasing
    ! root mass with depth which. ORCHIDEE uses the structural root profile to
    ! calculate rain infiltration along the roots, som inputs at depth and the
    ! root temperature for autotrophic respiration.

    ! NOTE: the shape of the profile is determined by its depth and the parameter
    ! humcste. For the moment humcste is PFT-dependent but constant over time. The
    ! depth is PFT-dependent (see above) and is a function of the active layer
    ! thickness when the soil discretization is used. The depth of the root profile
    ! could/should be made a function of tree diameter or plant biomass.
    root_profile(:, :, :, istruc) = zero
    DO jv = 1, nvm
      DO ji = 1, kjpindex

        ! Note that the integration will start at the top of the second layer (the
        ! top of the first layer is zdr(0) hence the zdr(1) is the top of the
        ! second layer) and will continue until the bottom of the profile. The
        ! bottom is the profile is calculated above but never extends deeper than
        ! the depth used in hydrol.f90 (in thermosoil.f90 the profile extends much
        ! deeper. The first layer is excluded because the profile will be used to
        ! extract water from the soil. The first layer is very thin and by extracting
        ! water it could dry to quickly. Also in reality not too many roots are found
        ! in the top mm of the soil. zdr describes the nodes and interfaces of the soil
        ! layers as proposed by de Rosnay 1999 (PhD thesis).
        z_top = zdr(1)
        z_bottom = zdr(root_depth(ji, jv, iinterface))

        ! Calculate the total surface area under an exponential curve between
        ! zdr(1) and the zdr(nslm)
        rpc = un / (EXP(- z_top * humcste(jv)) - EXP(- z_bottom * humcste(jv)))

          DO jsl = 2, root_depth(ji, jv, iinterface)

          ! Calculate the share of the total roots for layers which are "centered"
          ! at the nodes of the hydrology scheme. Centered was written in quotes
          ! because the layer is not symmetric. Using the nodes as the center of the
          ! layers follows De Rosnay (PhD, figure C.2 page 156). The root profile
          ! starts at the top of the second layer, ends at the bottom of the 11th
          ! layer and is calculated for the nodes in between.
          ! The following equation are derived (but rewritten) from the integrals
          ! of equations C9 to C11 of De Rosnay's (1999) PhD thesis (page 158).
          root_profile(ji, jv, jsl, istruc) = rpc * (EXP(- zdr(jsl - 1) * humcste(jv)) - EXP(- zdr(jsl) * humcste(jv)))

        END DO
        ! root_depth

        ! Top layer does not contain structural roots (see z_top)
        ! This line is not needed but was added as a reminder.
        root_profile(ji, jv, 1, istruc) = zero

          ! Error checking. Each root profile should add up to 1.
          IF (err_act .GT. 1) THEN
          IF (ABS(SUM(root_profile(ji, jv, :, istruc)) - un) .GT. 100 * EPSILON(un)) THEN
            WRITE(numout, *) 'pixel, PFT, sum of structural root_profile, ', ji, jv, SUM(root_profile(ji, jv, :, istruc))
            WRITE(numout, *) 'hydrol_root_profile, structural root_profile, ', root_profile(ji, jv, :, istruc)
            CALL ipslerr_p(plev, 'hydrol.f90', 'structural root profile does not add up to 1', 'Check its calculation', '')
          END IF
        END IF

      END DO
      ! kjpindex
    END DO
    ! nvm


    !! 3. Functional root profile

    ! Calculates the share of the root biomass in each soil layer based on
    ! the soil water content in each layer. The roots now follow the water.
    ! this results in a very dynamic root profile that may change every half
    ! hour (i.e. the time step for hydrol). ORCHIDEE uses the root function
    ! to calculate plant water stress (hydraulic_arch.f90) and the soil layers
    ! from which the transpiration is taken (hydrol.f90).

    ! NOTE: yet anonther root function is nutrient uptake. A separate root
    ! profile could be calculated to be used in the calculation of N uptake
    ! is stomate_soilcarbon.f90 (nitrogen_dynamics).
    root_profile(:, :, :, ifunc) = zero
    DO jv = 2, nvm
      DO ji = 1, kjpindex

        ! Plant available soil water per layer.
        ! The calculations could make use of sm and smw. These variables has
        ! been labelled soil moisture in kg/m2 and soil moisture at each layer
        ! at wilting point also in kg/m2. As an alternative swc and mcr could
        ! be used. swc is calculated in hydrol.f90 based on mc (m3/m3). mc is
        ! used to calculate smt (total soil water thus liquid + ice) and mcl is
        ! used to calculate sm (liquid only). sm denotes only liquid water, swc
        ! denotes liquid and frozen water. Given the focus on root function, a
        ! variable describing the liquid water seems the better choice.

        !+++CHECK+++
        ! The most important seems to be the difference in the dimensions of the
        ! variables: the dimensions of sm variables are kjpindex,nslm the
        ! dimension of the swc variable is kjpindex,nslm,nst. For the application
        ! we have in mind a different root profile for each soil tile (nst) seems
        ! desirable. Note that the difference in dimensions reflect the spatial
        ! scale of the model but it is not clear at which scale one approach
        ! (tile vs pixel) would be really prefered above another. Should we use
        ! the commented code further below?
        root_profile_tmp(:) = zero
        DO jsl = 2, root_depth(ji, jv, iinterface)
          root_profile_tmp(jsl) = MAX(zero, sm(ji, jsl) - smw(ji, jsl))
        END DO
        !+++++++++++

          ! Normalize to obtain a root profile (fraction between 0 and 1)
          IF (SUM(root_profile_tmp(:)) .GT. min_sechiba) THEN
          root_profile(ji, jv, :, ifunc) = root_profile_tmp(:) / SUM(root_profile_tmp(:))
        ELSE
          root_profile(ji, jv, :, ifunc) = zero
        END IF

        ! Top layer is not used for water uptake
        ! This line is not needed but was added as a reminder.
        root_profile(ji, jv, 1, ifunc) = zero

          ! The functional profile should also equal to one if there is soil moisture.
          IF (err_act .GT. 1) THEN
          IF (ABS(SUM(root_profile(ji, jv, :, ifunc)) - un) .GT. 100 * EPSILON(un) .AND. SUM(root_profile_tmp(:)) .GT. &
&min_sechiba) THEN
            WRITE(numout, *) 'pixel, PFT, sum of functional root_profile, ', ji, jv, SUM(root_profile(ji, jv, :, ifunc))
            WRITE(numout, *) 'hydrol_root_profile, functional root_profile, ', root_profile(ji, jv, :, ifunc)
            CALL ipslerr_p(plev, 'hydrol.f90', 'functional root profile does not add up to 1', 'Check its calculation', '')
          END IF
        END IF

      END DO
      ! kjpindex
    END DO
    ! ivm


    !+++CHECK+++
    ! The code above gives one root profile per pixel. The approach below would
    ! enable calculating one root profile per soil tile (bare, short and
    ! tall vegetation). NOTE that the variable name needs to be changed
    ! from root_dens to root_profile and that the dimensions need to re-
    ! ordered.
    !!$    DO ivm = 1, nvm
    !!$
    !!$       ! Link the pft to the soil tile
    !!$       istm = pref_soil_veg(ivm)
    !!$       IF ( is_tree(ivm) ) THEN
    !!$
    !!$          nroot_tmp(:) = zero
    !!$          ! Plant available soil water per layer. mcr is the residual soil
    !!$          ! water and depends on the soil type which is stored in njsc(ipts)
    !!$          DO ibdl = 2, nslm
    !!$             nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
    !!$          ENDDO
    !!$
    !!$       ELSE
    !!$
    !!$          ! Specific case for grasses where we only consider the first 1m of soil.
    !!$          ! Plant available soil water per layer. mcr is the residual soil
    !!$          ! water and depends on the soil type which is stored in njsc(ipts)
    !!$          nroot_tmp(:) = zero
    !!$          DO ibdl = 2, nslm
    !!$             IF (znt(ibdl) .LT. un) THEN
    !!$                nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
    !!$             ELSE
    !!$                nroot_tmp(ibdl) = zero
    !!$             END IF
    !!$          ENDDO
    !!$
    !!$       END IF
    !!$
    !!$       ! Normalize to obtain a root profile (fraction between 0 and 1)
    !!$       IF (SUM(nroot_tmp(:)) .GT. min_sechiba ) THEN
    !!$          root_dens(ipts,:,ivm) = nroot_tmp(:)/SUM(nroot_tmp(:))
    !!$       ELSE
    !!$          root_dens(ipts,:,ivm) = zero
    !!$       END IF
    !!$       root_dens(ipts,1,ivm) = zero
    !!$
    !!$    ENDDO
    !+++++++++++


  END SUBROUTINE hydrol_root_profile


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_coef
  !!
  !>\BRIEF        Computes coef for the linearised hydraulic conductivity
  !! k_lin=a_lin mc_lin+b_lin and the linearised diffusivity d_lin.
  !!
  !! DESCRIPTION  :
  !! First, we identify the interval i in which the current value of mc is located.
  !! Then, we give the values of the linearized parameters to compute
  !! conductivity and diffusivity as K=a*mc+b and d.
  !!
  !! RECENT CHANGE(S) : Addition of the dependence to profil_froz_hydro_ns
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_coef

  SUBROUTINE hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, ins, njsc)
    !$ACC ROUTINE SEQ

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: mc_used
    !! Used liquid water content
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x

    !_
    !& 
!& ================================================================================================================================

    IF (ok_freeze_cwrr) THEN

        ! Calculation of liquid and frozen saturation degrees with respect to residual
        ! x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
        ! 1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)

        DO jsl = 1, nslm

        x = 1._r_std - profil_froz_hydro_ns(ji, jsl, ins)

        ! mc_used is used in the calculation of hydrological properties
        ! It corresponds to a liquid mc, but the expression is different from mcl in hydrol_soil,
        ! to ensure that we get the a, b, d of the first bin when mcl<mcr
        mc_used = mcr(ji) + x * MAX((mc(ji, jsl, ins) - mcr(ji)), zero)
        !
        ! calcul de k based on mc_liq
        !
        i = MAX(imin, MIN(imax - 1, INT(imin + (imax - imin) * (mc_used - mcr(ji)) / (mcs(ji) - mcr(ji)))))
        a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm^2/d
        k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc_used + b_lin(i, jsl, ji))
        ! in mm/d
        ! loop on grid
      END DO

    ELSE
      ! .NOT. ok_freeze_cwrr
        DO jsl = 1, nslm

        ! it is impossible to consider a mc<mcr for the binning
        mc_ratio = MAX(mc(ji, jsl, ins) - mcr(ji), zero) / (mcs(ji) - mcr(ji))

        i = MAX(MIN(INT((imax - imin) * mc_ratio) + imin, imax - 1), imin)
        a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm^2/d
        k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc(ji, jsl, ins) + b_lin(i, jsl, &
&ji))
        ! in mm/d
      END DO
    END IF

  END SUBROUTINE hydrol_soil_coef_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_coef
    !!
    !>\BRIEF        Computes coef for the linearised hydraulic conductivity
    !! k_lin=a_lin mc_lin+b_lin and the linearised diffusivity d_lin.
    !!
    !! DESCRIPTION  :
    !! First, we identify the interval i in which the current value of mc is located.
    !! Then, we give the values of the linearized parameters to compute
    !! conductivity and diffusivity as K=a*mc+b and d.
    !!
    !! RECENT CHANGE(S) : Addition of the dependence to profil_froz_hydro_ns
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_coef

    SUBROUTINE hydrol_soil_coef(mcr, mcs, kjpindex, ins, njsc)

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: mc_used
    !! Used liquid water content
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x

    !_
    !& 
!& ================================================================================================================================

    IF (ok_freeze_cwrr) THEN

        ! Calculation of liquid and frozen saturation degrees with respect to residual
        ! x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
        ! 1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)

        DO jsl = 1, nslm
        DO ji = 1, kjpindex

          x = 1._r_std - profil_froz_hydro_ns(ji, jsl, ins)

          ! mc_used is used in the calculation of hydrological properties
          ! It corresponds to a liquid mc, but the expression is different from mcl in hydrol_soil,
          ! to ensure that we get the a, b, d of the first bin when mcl<mcr
          mc_used = mcr(ji) + x * MAX((mc(ji, jsl, ins) - mcr(ji)), zero)
          !
          ! calcul de k based on mc_liq
          !
          i = MAX(imin, MIN(imax - 1, INT(imin + (imax - imin) * (mc_used - mcr(ji)) / (mcs(ji) - mcr(ji)))))
          a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm^2/d
          k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc_used + b_lin(i, jsl, ji))
          ! in mm/d
        END DO
        ! loop on grid
      END DO

    ELSE
      ! .NOT. ok_freeze_cwrr
        DO jsl = 1, nslm
        DO ji = 1, kjpindex

          ! it is impossible to consider a mc<mcr for the binning
          mc_ratio = MAX(mc(ji, jsl, ins) - mcr(ji), zero) / (mcs(ji) - mcr(ji))

          i = MAX(MIN(INT((imax - imin) * mc_ratio) + imin, imax - 1), imin)
          a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm^2/d
          k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc(ji, jsl, ins) + b_lin(i, &
&jsl, ji))
          ! in mm/d
        END DO
      END DO
    END IF

  END SUBROUTINE hydrol_soil_coef


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_smooth_under_mcr
  !!
  !>\BRIEF        : Modifies the soil moisture profile to avoid under-residual values,
  !!                then diagnoses the points where such "excess" values remain.
  !!
  !! DESCRIPTION  :
  !! The "excesses" under-residual are corrected from top to bottom, by transfer of excesses
  !! to the lower layers. The reverse transfer is performed to smooth any remaining "excess" in the bottom layer.
  !! If some "excess" remain afterwards, the entire soil profile is at the threshold value (mcs or mcr),
  !! and the remaining "excess" is necessarily concentrated in the top layer.
  !! This allowing diagnosing the flag is_under_mcr.
  !! Eventually, the remaining "excess" is split over the entire profile
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. Smoothes the profile to avoid negative values of punctual soil moisture
  !! Note that we check that mc > min_sechiba in hydrol_soil
  !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
  !!    and export the difference with the flux
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_smooth_under_mcr

  SUBROUTINE hydrol_soil_smooth_under_mcr_acc(ji, mcr, kjpindex, ins, njsc, is_under_mcr, check)
    !$ACC ROUTINE SEQ

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(nstm), INTENT(OUT) :: is_under_mcr
    !! Flag diagnosing under residual soil moisture
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: excess
    REAL(KIND = r_std) :: excessji
    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine

    !_
    !& 
!& ================================================================================================================================

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

      !! 2. Smoothes the profile to avoid negative values of punctual soil moisture

      ! 2.1 smoothing from top to bottom
      DO jsl = 1, nslm - 2
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl + 1) + dz(jsl + 2))
    END DO

    jsl = nslm - 1
    excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
    mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
    mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / dz(jsl + 1)

    jsl = nslm
    excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
    mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
    mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * dz(jsl) / (dz(jsl - 1) + dz(jsl))

      ! 2.2 smoothing from bottom to top
      DO jsl = nslm - 1, 2, - 1
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl - 1) + dz(jsl))
    END DO

    ! 2.3 diagnoses is_under_mcr(ji), and updates the entire profile
    ! excess > 0
    excessji = mask_soiltile(ji, ins) * MAX(mcr(ji) - mc(ji, 1, ins), zero)
    mc(ji, 1, ins) = mc(ji, 1, ins) + excessji
    ! then mc(1)=mcr
    is_under_mcr(ins) = (excessji .GT. min_sechiba)

      ! 2.4 The amount of water corresponding to excess in the top soil layer is redistributed in all soil layers
      ! -excess(ji) * dz(2) / deux donne le deficit total, negatif, en mm
      ! diviser par la profondeur totale en mm donne des delta_mc identiques en chaque couche, en mm
      ! retransformes en delta_mm par couche selon les bonnes eqs (eqs_hydrol.pdf, Eqs 13-15), puis sommes
      ! retourne bien le deficit total en mm
      DO jsl = 1, nslm
      mc(ji, jsl, ins) = mc(ji, jsl, ins) - excessji * dz(2) / (deux * zmaxh * mille)
    END DO
    ! This can lead to mc(jsl) < mcr depending on the value of excess,
      ! but this is no major pb for the diffusion
      ! Yet, we need to prevent evaporation if is_under_mcr

      !! Note that we check that mc > min_sechiba in hydrol_soil

      ! We just make sure that mc remains at 0 where soiltile=0
      DO jsl = 1, nslm
      mc(ji, jsl, ins) = mask_soiltile(ji, ins) * mc(ji, jsl, ins)
    END DO

      !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci since we just redistribute the deficit
      check(ins) = tmcf - tmci
    END IF

  END SUBROUTINE hydrol_soil_smooth_under_mcr_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_smooth_under_mcr
    !!
    !>\BRIEF        : Modifies the soil moisture profile to avoid under-residual values,
    !!                then diagnoses the points where such "excess" values remain.
    !!
    !! DESCRIPTION  :
    !! The "excesses" under-residual are corrected from top to bottom, by transfer of excesses
    !! to the lower layers. The reverse transfer is performed to smooth any remaining "excess" in the bottom layer.
    !! If some "excess" remain afterwards, the entire soil profile is at the threshold value (mcs or mcr),
    !! and the remaining "excess" is necessarily concentrated in the top layer.
    !! This allowing diagnosing the flag is_under_mcr.
    !! Eventually, the remaining "excess" is split over the entire profile
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. Smoothes the profile to avoid negative values of punctual soil moisture
    !! Note that we check that mc > min_sechiba in hydrol_soil
    !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
    !!    and export the difference with the flux
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_smooth_under_mcr

    SUBROUTINE hydrol_soil_smooth_under_mcr(mcr, kjpindex, ins, njsc, is_under_mcr, check)

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex, nstm), INTENT(OUT) :: is_under_mcr
    !! Flag diagnosing under residual soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std) :: excess
    REAL(KIND = r_std), DIMENSION(kjpindex) :: excessji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine

    !_
    !& 
!& ================================================================================================================================

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. Smoothes the profile to avoid negative values of punctual soil moisture

      ! 2.1 smoothing from top to bottom
      DO jsl = 1, nslm - 2
      DO ji = 1, kjpindex
        excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
        mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl + 1) + dz(jsl + 2))
      END DO
    END DO

    jsl = nslm - 1
    DO ji = 1, kjpindex
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / dz(jsl + 1)
    END DO

    jsl = nslm
    DO ji = 1, kjpindex
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * dz(jsl) / (dz(jsl - 1) + dz(jsl))
    END DO

      ! 2.2 smoothing from bottom to top
      DO jsl = nslm - 1, 2, - 1
      DO ji = 1, kjpindex
        excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
        mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl - 1) + dz(jsl))
      END DO
    END DO

      ! 2.3 diagnoses is_under_mcr(ji), and updates the entire profile
      ! excess > 0
      DO ji = 1, kjpindex
      excessji(ji) = mask_soiltile(ji, ins) * MAX(mcr(ji) - mc(ji, 1, ins), zero)
    END DO
    DO ji = 1, kjpindex
      mc(ji, 1, ins) = mc(ji, 1, ins) + excessji(ji)
      ! then mc(1)=mcr
      is_under_mcr(ji, ins) = (excessji(ji) .GT. min_sechiba)
    END DO

      ! 2.4 The amount of water corresponding to excess in the top soil layer is redistributed in all soil layers
      ! -excess(ji) * dz(2) / deux donne le deficit total, negatif, en mm
      ! diviser par la profondeur totale en mm donne des delta_mc identiques en chaque couche, en mm
      ! retransformes en delta_mm par couche selon les bonnes eqs (eqs_hydrol.pdf, Eqs 13-15), puis sommes
      ! retourne bien le deficit total en mm
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        mc(ji, jsl, ins) = mc(ji, jsl, ins) - excessji(ji) * dz(2) / (deux * zmaxh * mille)
      END DO
    END DO
    ! This can lead to mc(jsl) < mcr depending on the value of excess,
      ! but this is no major pb for the diffusion
      ! Yet, we need to prevent evaporation if is_under_mcr

      !! Note that we check that mc > min_sechiba in hydrol_soil

      ! We just make sure that mc remains at 0 where soiltile=0
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        mc(ji, jsl, ins) = mask_soiltile(ji, ins) * mc(ji, jsl, ins)
      END DO
    END DO

      !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci since we just redistribute the deficit
      check(:, ins) = tmcf(:) - tmci(:)
    END IF

  END SUBROUTINE hydrol_soil_smooth_under_mcr

  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_froz
  !!
  !>\BRIEF        Computes profil_froz_hydro_ns, the fraction of frozen water in the soil layers.
  !!
  !! DESCRIPTION  :
  !!
  !! RECENT CHANGE(S) : Created by A. Ducharne in 2016.
  !!
  !! MAIN OUTPUT VARIABLE(S) : profil_froz_hydro_ns
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_froz

  SUBROUTINE hydrol_soil_froz_acc(ji, nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)
    !$ACC ROUTINE SEQ

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x
    REAL(KIND = r_std) :: denom
    REAL(KIND = r_std) :: froz_frac_moy
    REAL(KIND = r_std) :: smtot_moy
    REAL(KIND = r_std), DIMENSION(nslm) :: mc_ns

    !_
    !& 
!& ================================================================================================================================

    !    ONLY FOR THE (ok_freeze_cwrr) CASE

    ! Calculation of liquid and frozen saturation degrees above residual moisture
    !   x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
    !   1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)
    ! It's important for the good work of the water diffusion scheme (tridiag) that the total
    ! liquid water also includes mcr, so mcl > 0 even when x=0

    DO jsl = 1, nslm
      ! Van Genuchten parameter for thermodynamical calculation
      m = 1. - 1. / nvan(ji)

        IF ((.NOT. ok_thermodynamical_freezing) .OR. (mc(ji, jsl, ins) .LT. (mcr(ji) + min_sechiba))) THEN
        ! Linear soil freezing or soil moisture below residual
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          x = (stempdiag(ji, jsl) - (fr_center - fr_dT / 2.)) / fr_dT
        ELSE
          x = 0._r_std
        END IF
      ELSE IF (ok_thermodynamical_freezing) THEN
        ! Thermodynamical soil freezing
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          ! Factor 2.2 from the PhD of Isabelle Gouttevin
          x = MIN(((mcs(ji) - mcr(ji)) * ((2.2 * 1000. * avan(ji) * (fr_center + fr_dT / 2. - stempdiag(ji, jsl)) * lhf / &
&ZeroCelsius / 10.) ** nvan(ji) + 1.) ** (- m)) / (mc(ji, jsl, ins) - mcr(ji)), 1._r_std)
        ELSE
          x = 0._r_std
        END IF
      END IF

      profil_froz_hydro_ns(ji, jsl, ins) = 1._r_std - x

      mc_ns(jsl) = mc(ji, jsl, ins) / mcs(ji)

      ! loop on grid
    END DO

    ! Applay correction on the frozen fraction
    ! Depends on two external parameters: froz_frac_corr and smtot_corr
    froz_frac_moy = zero
    denom = zero
    DO jsl = 1, nslm
      froz_frac_moy = froz_frac_moy + dh(jsl) * profil_froz_hydro_ns(ji, jsl, ins)
      denom = denom + dh(jsl)
    END DO
    froz_frac_moy = froz_frac_moy / denom

    smtot_moy = zero
    denom = zero
    DO jsl = 1, nslm - 1
      smtot_moy = smtot_moy + dh(jsl) * mc_ns(jsl)
      denom = denom + dh(jsl)
    END DO
    smtot_moy = smtot_moy / denom

      DO jsl = 1, nslm
      profil_froz_hydro_ns(ji, jsl, ins) = MIN(profil_froz_hydro_ns(ji, jsl, ins) * (froz_frac_moy ** froz_frac_corr) * (smtot_moy &
&** smtot_corr), max_froz_hydro)
    END DO

  END SUBROUTINE hydrol_soil_froz_acc

    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_froz
    !!
    !>\BRIEF        Computes profil_froz_hydro_ns, the fraction of frozen water in the soil layers.
    !!
    !! DESCRIPTION  :
    !!
    !! RECENT CHANGE(S) : Created by A. Ducharne in 2016.
    !!
    !! MAIN OUTPUT VARIABLE(S) : profil_froz_hydro_ns
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_froz

    SUBROUTINE hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x
    REAL(KIND = r_std) :: denom
    REAL(KIND = r_std), DIMENSION(kjpindex) :: froz_frac_moy
    REAL(KIND = r_std), DIMENSION(kjpindex) :: smtot_moy
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mc_ns

    !_
    !& 
!& ================================================================================================================================

    !    ONLY FOR THE (ok_freeze_cwrr) CASE

    ! Calculation of liquid and frozen saturation degrees above residual moisture
    !   x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
    !   1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)
    ! It's important for the good work of the water diffusion scheme (tridiag) that the total
    ! liquid water also includes mcr, so mcl > 0 even when x=0

    DO jsl = 1, nslm
      DO ji = 1, kjpindex
        ! Van Genuchten parameter for thermodynamical calculation
        m = 1. - 1. / nvan(ji)

          IF ((.NOT. ok_thermodynamical_freezing) .OR. (mc(ji, jsl, ins) .LT. (mcr(ji) + min_sechiba))) THEN
          ! Linear soil freezing or soil moisture below residual
            IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
            x = 1._r_std
          ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
            x = (stempdiag(ji, jsl) - (fr_center - fr_dT / 2.)) / fr_dT
          ELSE
            x = 0._r_std
          END IF
        ELSE IF (ok_thermodynamical_freezing) THEN
          ! Thermodynamical soil freezing
            IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
            x = 1._r_std
          ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
            ! Factor 2.2 from the PhD of Isabelle Gouttevin
            x = MIN(((mcs(ji) - mcr(ji)) * ((2.2 * 1000. * avan(ji) * (fr_center + fr_dT / 2. - stempdiag(ji, jsl)) * lhf / &
&ZeroCelsius / 10.) ** nvan(ji) + 1.) ** (- m)) / (mc(ji, jsl, ins) - mcr(ji)), 1._r_std)
          ELSE
            x = 0._r_std
          END IF
        END IF

        profil_froz_hydro_ns(ji, jsl, ins) = 1._r_std - x

        mc_ns(ji, jsl) = mc(ji, jsl, ins) / mcs(ji)

      END DO
      ! loop on grid
    END DO

    ! Applay correction on the frozen fraction
    ! Depends on two external parameters: froz_frac_corr and smtot_corr
    froz_frac_moy(:) = zero
    denom = zero
    DO jsl = 1, nslm
      froz_frac_moy(:) = froz_frac_moy(:) + dh(jsl) * profil_froz_hydro_ns(:, jsl, ins)
      denom = denom + dh(jsl)
    END DO
    froz_frac_moy(:) = froz_frac_moy(:) / denom

    smtot_moy(:) = zero
    denom = zero
    DO jsl = 1, nslm - 1
      smtot_moy(:) = smtot_moy(:) + dh(jsl) * mc_ns(:, jsl)
      denom = denom + dh(jsl)
    END DO
    smtot_moy(:) = smtot_moy(:) / denom

      DO jsl = 1, nslm
      profil_froz_hydro_ns(:, jsl, ins) = MIN(profil_froz_hydro_ns(:, jsl, ins) * (froz_frac_moy(:) ** froz_frac_corr) * &
&(smtot_moy(:) ** smtot_corr), max_froz_hydro)
    END DO

  END SUBROUTINE hydrol_soil_froz


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_infilt
  !!
  !>\BRIEF        Infiltration
  !!
  !! DESCRIPTION  :
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. Infiltration process
  !! 2.1 Initialization of time counter and infiltration rate
  !! 2.2 Infiltration layer by layer, accounting for an exponential law for subgrid variability
  !! 2.3 Resulting infiltration and surface runoff
  !! 3. For water conservation check, we calculate the total SM at the beginning of the routine,
  !!    and export the difference with the flux
  !! 5. Local verification
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !! Adding checks and interactions variables with hydrol_soil, but the processes are unchanged
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_infilt

  SUBROUTINE hydrol_soil_infilt_acc(error_flag_hydrol_soil_infilt_1, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, &
&flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
    !$ACC ROUTINE SEQ

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! GLOBAL (in or inout)
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_soil_infilt_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell
    !!  (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), INTENT(IN) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: check
    !! delta SM - flux (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: ru_infilt
    !! Surface runoff from soil_infilt (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: qinfilt_ns
    !! Effective infiltration flux (mm/dt_sechiba)

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    !! Indices
    REAL(KIND = r_std) :: wat_inf_pot
    !! infiltrable water in the layer
    REAL(KIND = r_std) :: wat_inf
    !! infiltrated water in the layer
    REAL(KIND = r_std) :: dt_tmp
    !! time remaining before the end of the time step
    REAL(KIND = r_std) :: dt_inf
    !! the time it takes to complete the infiltration in the
    !! layer
    REAL(KIND = r_std) :: k_m
    !! the mean conductivity used for the saturated front
    REAL(KIND = r_std) :: infilt_tmp
    !! infiltration rate for the considered layer
    REAL(KIND = r_std) :: infilt_tot
    !! total infiltration
    REAL(KIND = r_std) :: flux_tmp
    !! rate at which precip hits the ground

    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine (kg/m2)
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine (kg/m2)


    !_
    !& 
!& ================================================================================================================================

    ! If data (or coupling with GCM) was available, a parameterization for subgrid rainfall could be performed

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

    !! 2. Infiltration process

    !! 2.1 Initialization

    !! First we fill up the first layer (about 1mm) without any resistance and quasi-immediately
    wat_inf_pot = MAX((mcs(ji) - mc(ji, 1, ins)) * dz(2) / deux, zero)
    wat_inf = MIN(wat_inf_pot, flux_infilt)
    mc(ji, 1, ins) = mc(ji, 1, ins) + wat_inf * deux / dz(2)
    !

    !! Initialize a countdown for infiltration during the time-step and the value of potential runoff
    dt_tmp = dt_sechiba / one_day
    infilt_tot = wat_inf
    !! Compute the rate at which water will try to infiltrate each layer
    ! flux_temp is converted here to the same unit as k_m
    flux_tmp = (flux_infilt - wat_inf) / dt_tmp

      !! 2.2 Infiltration layer by layer
      DO jsl = 2, nslm - 1
      !! Infiltrability of each layer if under a saturated one
      ! This is computed by an simple arithmetic average because
      ! the time step (30min) is not appropriate for a geometric average (advised by Haverkamp and Vauclin)
      k_m = (k(ji, jsl) + ks(ji) * kfact(jsl - 1, ji) * kfact_root(ji, jsl, ins)) / deux

        IF (ok_freeze_cwrr) THEN
        IF (stempdiag(ji, jsl) .LT. ZeroCelsius) THEN
          k_m = k(ji, jsl)
        END IF
      END IF

      !! We compute the mean rate at which water actually infiltrate:
      ! Subgrid: Exponential distribution of k around k_m, but average p directly used
      ! See d'Orgeval 2006, p 78, but it's not fully clear to me (AD16***)
      infilt_tmp = k_m * (un - EXP(- flux_tmp / k_m))

      !! From which we deduce the time it takes to fill up the layer or to end the time step...
      wat_inf_pot = MAX((mcs(ji) - mc(ji, jsl, ins)) * (dz(jsl) + dz(jsl + 1)) / deux, zero)
      IF (infilt_tmp > min_sechiba) THEN
        dt_inf = MIN(wat_inf_pot / infilt_tmp, dt_tmp)
        ! The water infiltration TIME has to limited by what is still available for infiltration.
          IF (dt_inf * infilt_tmp > flux_infilt - infilt_tot) THEN
          dt_inf = MAX(flux_infilt - infilt_tot, zero) / infilt_tmp
        END IF
      ELSE
        dt_inf = dt_tmp
      END IF

      !! The water enters in the layer
      wat_inf = dt_inf * infilt_tmp
      ! bviously the moisture content
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + wat_inf * deux / (dz(jsl) + dz(jsl + 1))
      ! the time remaining before the next time step
      dt_tmp = dt_tmp - dt_inf
      ! and finally the infilt_tot (which is just used to check if there is a problem, below)
      infilt_tot = infilt_tot + infilt_tmp * dt_inf
    END DO

    !! 2.3 Resulting infiltration and surface runoff
    ru_infilt(ins) = flux_infilt - infilt_tot
    qinfilt_ns(ins) = infilt_tot

      !! 3. For water conservation check: we calculate the total SM at the beginning of the routine
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci+infilt_tot
      check(ins) = tmcf - (tmci + infilt_tot)
    END IF

      !! 5. Local verification
      IF (infilt_tot .LT. - min_sechiba .OR. infilt_tot .GT. flux_infilt + min_sechiba) THEN
      error_flag_hydrol_soil_infilt_1 = error_flag_hydrol_soil_infilt_1 + 1
    END IF

  END SUBROUTINE hydrol_soil_infilt_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_infilt
    !!
    !>\BRIEF        Infiltration
    !!
    !! DESCRIPTION  :
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. Infiltration process
    !! 2.1 Initialization of time counter and infiltration rate
    !! 2.2 Infiltration layer by layer, accounting for an exponential law for subgrid variability
    !! 2.3 Resulting infiltration and surface runoff
    !! 3. For water conservation check, we calculate the total SM at the beginning of the routine,
    !!    and export the difference with the flux
    !! 5. Local verification
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !! Adding checks and interactions variables with hydrol_soil, but the processes are unchanged
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_infilt

    SUBROUTINE hydrol_soil_infilt(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, flux_infilt, stempdiag, qinfilt_ns, &
&ru_infilt, check)

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! GLOBAL (in or inout)
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell
    !!  (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: ru_infilt
    !! Surface runoff from soil_infilt (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: qinfilt_ns
    !! Effective infiltration flux (mm/dt_sechiba)

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    !! Indices
    REAL(KIND = r_std), DIMENSION(kjpindex) :: wat_inf_pot
    !! infiltrable water in the layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: wat_inf
    !! infiltrated water in the layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dt_tmp
    !! time remaining before the end of the time step
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dt_inf
    !! the time it takes to complete the infiltration in the
    !! layer
    REAL(KIND = r_std) :: k_m
    !! the mean conductivity used for the saturated front
    REAL(KIND = r_std), DIMENSION(kjpindex) :: infilt_tmp
    !! infiltration rate for the considered layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: infilt_tot
    !! total infiltration
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_tmp
    !! rate at which precip hits the ground

    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine (kg/m2)


    !_
    !& 
!& ================================================================================================================================

    ! If data (or coupling with GCM) was available, a parameterization for subgrid rainfall could be performed

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. Infiltration process

      !! 2.1 Initialization

      DO ji = 1, kjpindex
      !! First we fill up the first layer (about 1mm) without any resistance and quasi-immediately
      wat_inf_pot(ji) = MAX((mcs(ji) - mc(ji, 1, ins)) * dz(2) / deux, zero)
      wat_inf(ji) = MIN(wat_inf_pot(ji), flux_infilt(ji))
      mc(ji, 1, ins) = mc(ji, 1, ins) + wat_inf(ji) * deux / dz(2)
      !
    END DO

    !! Initialize a countdown for infiltration during the time-step and the value of potential runoff
    dt_tmp(:) = dt_sechiba / one_day
    infilt_tot(:) = wat_inf(:)
    !! Compute the rate at which water will try to infiltrate each layer
    ! flux_temp is converted here to the same unit as k_m
    flux_tmp(:) = (flux_infilt(:) - wat_inf(:)) / dt_tmp(:)

      !! 2.2 Infiltration layer by layer
      DO jsl = 2, nslm - 1
      DO ji = 1, kjpindex
        !! Infiltrability of each layer if under a saturated one
        ! This is computed by an simple arithmetic average because
        ! the time step (30min) is not appropriate for a geometric average (advised by Haverkamp and Vauclin)
        k_m = (k(ji, jsl) + ks(ji) * kfact(jsl - 1, ji) * kfact_root(ji, jsl, ins)) / deux

          IF (ok_freeze_cwrr) THEN
          IF (stempdiag(ji, jsl) .LT. ZeroCelsius) THEN
            k_m = k(ji, jsl)
          END IF
        END IF

        !! We compute the mean rate at which water actually infiltrate:
        ! Subgrid: Exponential distribution of k around k_m, but average p directly used
        ! See d'Orgeval 2006, p 78, but it's not fully clear to me (AD16***)
        infilt_tmp(ji) = k_m * (un - EXP(- flux_tmp(ji) / k_m))

        !! From which we deduce the time it takes to fill up the layer or to end the time step...
        wat_inf_pot(ji) = MAX((mcs(ji) - mc(ji, jsl, ins)) * (dz(jsl) + dz(jsl + 1)) / deux, zero)
        IF (infilt_tmp(ji) > min_sechiba) THEN
          dt_inf(ji) = MIN(wat_inf_pot(ji) / infilt_tmp(ji), dt_tmp(ji))
          ! The water infiltration TIME has to limited by what is still available for infiltration.
            IF (dt_inf(ji) * infilt_tmp(ji) > flux_infilt(ji) - infilt_tot(ji)) THEN
            dt_inf(ji) = MAX(flux_infilt(ji) - infilt_tot(ji), zero) / infilt_tmp(ji)
          END IF
        ELSE
          dt_inf(ji) = dt_tmp(ji)
        END IF

        !! The water enters in the layer
        wat_inf(ji) = dt_inf(ji) * infilt_tmp(ji)
        ! bviously the moisture content
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + wat_inf(ji) * deux / (dz(jsl) + dz(jsl + 1))
        ! the time remaining before the next time step
        dt_tmp(ji) = dt_tmp(ji) - dt_inf(ji)
        ! and finally the infilt_tot (which is just used to check if there is a problem, below)
        infilt_tot(ji) = infilt_tot(ji) + infilt_tmp(ji) * dt_inf(ji)
      END DO
    END DO

    !! 2.3 Resulting infiltration and surface runoff
    ru_infilt(:, ins) = flux_infilt(:) - infilt_tot(:)
    qinfilt_ns(:, ins) = infilt_tot(:)

      !! 3. For water conservation check: we calculate the total SM at the beginning of the routine
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci+infilt_tot
      check(:, ins) = tmcf(:) - (tmci(:) + infilt_tot(:))
    END IF

      !! 5. Local verification
      DO ji = 1, kjpindex
      IF (infilt_tot(ji) .LT. - min_sechiba .OR. infilt_tot(ji) .GT. flux_infilt(ji) + min_sechiba) THEN
        WRITE(numout, *) 'Error in the calculation of infilt tot', infilt_tot(ji)
        WRITE(numout, *) 'k, ji, jst, mc', k(ji, 1 : 2), ji, ins, mc(ji, 1, ins)
        CALL ipslerr_p(3, 'hydrol_soil_infilt', 'We will STOP now.', 'Error in calculation of infilt tot', '')
      END IF
    END DO

  END SUBROUTINE hydrol_soil_infilt


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_split_soil
  !!
  !>\BRIEF        Splits 2d variables into 3d variables, per soiltile (_ns suffix), at the beginning of hydrol
  !!              At this stage, the forcing fluxes to hydrol are transformed from grid-cell averages
  !!              to mean fluxes over vegtot=sum(soiltile)
  !!
  !! DESCRIPTION  :
  !! 1. Split 2d variables into 3d variables, per soiltile
  !! 1.1 Throughfall
  !! 1.2 Bare soil evaporation
  !! 1.2.2 ae_ns new
  !! 1.3 transpiration
  !! 1.4 root sink
  !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes
  !! 2.1 precisol
  !! 2.2 ae_ns and evapnu
  !! 2.3 transpiration
  !! 2.4 root sink
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne to match the simplification of hydrol_soil
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================


  SUBROUTINE hydrol_split_soil_acc(error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, &
&error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, &
&error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, &
&error_flag_hydrol_split_soil_12, ji, kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, &
&tot_bare_soil, us, e_frac, F_absorption)
    !$ACC ROUTINE SEQ

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_12
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_11
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_10
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_9
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_8
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_7
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_6
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_5
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_4
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_3
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_2
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max Vegetation map
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soiltile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: vevapnu
    !! Bare soil evaporation
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evap_bare_lim
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: evap_bare_lim_ns
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(IN) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Relative humidity per layer
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydraulic_arch = .TRUE.)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    REAL(KIND = r_std) :: tmp_check1
    REAL(KIND = r_std) :: tmp_check2
    REAL(KIND = r_std), DIMENSION(nstm) :: tmp_check3
    LOGICAL :: error
    !_
    !& 
!& ================================================================================================================================

    !! 1. Split 2d variables into 3d variables, per soiltile

    ! Reminders:
    !  corr_veg_soil(:,nvm,nstm) = PFT fraction per soiltile in each grid-cell
    !      corr_veg_soil(ji,jv,jst)=veget_max(ji,jv)/soiltile(ji,jst)
    !  soiltile(:,nstm) = fraction of vegtot covered by each soiltile (0-1, unitless)
    !  vegtot(:) = total fraction of grid-cell covered by PFTs (fraction with bare soil + vegetation)
    !  veget_max(:,nvm) = PFT fractions of vegtot+frac_nobio
    !  veget(:,nvm) =  fractions (of vegtot+frac_nobio) covered by vegetation in each PFT
    !       BUT veget(:,1)=veget_max(:,1)
    !  frac_bare(:,nvm) = fraction (of veget_max) with bare soil in each PFT
    !  tot_bare_soil(:) = fraction of grid mesh covered by all bare soil (=SUM(frac_bare*veget_max))
    !  frac_bare_ns(:,nstm) = evaporating bare soil fraction (of vegtot) per soiltile (defined in hydrol_vegupd)

    !! 1.1 Throughfall
    ! Transformation from precisol (flux from PFT jv in m2 of grid-mesh)
    ! to  precisol_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    precisol_ns(ji, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      IF ((veget_max(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
        precisol_ns(ji, jst) = precisol_ns(ji, jst) + precisol(ji, jv) / (soiltile(ji, jst) * vegtot(ji))
      END IF
    END DO


    !! 1.2 Bare soil evaporation and ae_ns
    ae_ns(ji, :) = zero
    DO jst = 1, nstm
      IF (evap_bare_lim(ji) .GT. min_sechiba) THEN
        ae_ns(ji, jst) = vevapnu(ji) * evap_bare_lim_ns(ji, jst) / evap_bare_lim(ji)
      END IF
    END DO

    !! 1.3 transpiration
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to tr_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    ! To do next: simplify the use of humrelv(ji,jv,jst) /humrel(ji,jv), since both are equal
    tr_ns(ji, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
        tr_ns(ji, jst) = tr_ns(ji, jst) + transpir(ji, jv) * (humrelv(ji, jv, jst) / humrel(ji, jv)) / (soiltile(ji, jst) * &
&vegtot(ji))

      END IF
    END DO

    !! 1.4 root sink
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to root_sink (flux from contributing PFTs and soil layer with another unit, in m2 of soiltile)
    rootsink(ji, :, :) = zero

      IF (ok_hydrol_arch) THEN

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        ! OBS jst = 1,nstm
          DO jsl = 1, nslm
          IF (humrel(ji, jv) .GT. min_sechiba .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
            IF (is_tuzet_hydrol_arch) THEN
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (F_absorption(ji, jv) * e_frac(ji, jv, jsl, jst) * dt_sechiba * &
&kilo_to_unit)
            ELSE
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (transpir(ji, jv) * e_frac(ji, jv, jsl, jst)) / (soiltile(ji, jst) &
&* vegtot(ji))
            END IF
          END IF
        END DO
      END DO

    ELSE

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        DO jsl = 1, nslm
          IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
            rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + transpir(ji, jv) * (us(ji, jv, jst, jsl) / humrel(ji, jv)) / &
&(soiltile(ji, jst) * vegtot(ji))
            ! rootsink(ji,1,jst)=0 as us(ji,jv,jst,1)=0
          END IF
        END DO
      END DO

    END IF
    ! ok_hydrol_arch

      !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes (grid-cell average)

      IF (check_cwrr) THEN

      error = .FALSE.

      !! 2.1 precisol

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + precisol_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO

      tmp_check2 = zero
      DO jv = 1, nvm
        tmp_check2 = tmp_check2 + precisol(ji, jv)
      END DO

        IF (ABS(tmp_check1 - tmp_check2) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_1 = error_flag_hydrol_split_soil_1 + 1
        DO jv = 1, nvm
          error_flag_hydrol_split_soil_2 = error_flag_hydrol_split_soil_2 + 1
        END DO
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_3 = error_flag_hydrol_split_soil_3 + 1
        END DO
        error = .TRUE.
      END IF

      !! 2.2 ae_ns and evapnu

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + ae_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO


        IF (ABS(tmp_check1 - vevapnu(ji)) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_4 = error_flag_hydrol_split_soil_4 + 1
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_5 = error_flag_hydrol_split_soil_5 + 1
        END DO
        error = .TRUE.
      END IF

      !! 2.3 transpiration

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + tr_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO

      tmp_check2 = zero
      DO jv = 1, nvm
        tmp_check2 = tmp_check2 + transpir(ji, jv)
      END DO

        IF (ABS(tmp_check1 - tmp_check2) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_6 = error_flag_hydrol_split_soil_6 + 1
        DO jv = 1, nvm
          error_flag_hydrol_split_soil_7 = error_flag_hydrol_split_soil_7 + 1
          DO jst = 1, nstm
            error_flag_hydrol_split_soil_8 = error_flag_hydrol_split_soil_8 + 1
          END DO
        END DO
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_9 = error_flag_hydrol_split_soil_9 + 1
        END DO
        error = .TRUE.
      END IF


      !! 2.4 root sink

      tmp_check3(:) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          tmp_check3(jst) = tmp_check3(jst) + rootsink(ji, jsl, jst)
        END DO
      END DO

        DO jst = 1, nstm
        IF (ABS(tmp_check3(jst) - tr_ns(ji, jst)) .GT. allowed_err) THEN
          error_flag_hydrol_split_soil_10 = error_flag_hydrol_split_soil_10 + 1
          DO jv = 1, nvm
            error_flag_hydrol_split_soil_11 = error_flag_hydrol_split_soil_11 + 1
          END DO
          error = .TRUE.
        END IF
      END DO


        !! Exit if error was found previously in this subroutine
        IF (error) THEN
        error_flag_hydrol_split_soil_12 = error_flag_hydrol_split_soil_12 + 1
      END IF

    END IF
    ! end of check_cwrr


  END SUBROUTINE hydrol_split_soil_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_split_soil
    !!
    !>\BRIEF        Splits 2d variables into 3d variables, per soiltile (_ns suffix), at the beginning of hydrol
    !!              At this stage, the forcing fluxes to hydrol are transformed from grid-cell averages
    !!              to mean fluxes over vegtot=sum(soiltile)
    !!
    !! DESCRIPTION  :
    !! 1. Split 2d variables into 3d variables, per soiltile
    !! 1.1 Throughfall
    !! 1.2 Bare soil evaporation
    !! 1.2.2 ae_ns new
    !! 1.3 transpiration
    !! 1.4 root sink
    !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes
    !! 2.1 precisol
    !! 2.2 ae_ns and evapnu
    !! 2.3 transpiration
    !! 2.4 root sink
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne to match the simplification of hydrol_soil
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================


    SUBROUTINE hydrol_split_soil(kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, &
&tot_bare_soil, us, e_frac, F_absorption)

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max Vegetation map
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soiltile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: vevapnu
    !! Bare soil evaporation
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evap_bare_lim
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: evap_bare_lim_ns
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(IN) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Relative humidity per layer
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydraulic_arch = .TRUE.)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmp_check1
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmp_check2
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: tmp_check3
    LOGICAL :: error
    !_
    !& 
!& ================================================================================================================================

    !! 1. Split 2d variables into 3d variables, per soiltile

    ! Reminders:
    !  corr_veg_soil(:,nvm,nstm) = PFT fraction per soiltile in each grid-cell
    !      corr_veg_soil(ji,jv,jst)=veget_max(ji,jv)/soiltile(ji,jst)
    !  soiltile(:,nstm) = fraction of vegtot covered by each soiltile (0-1, unitless)
    !  vegtot(:) = total fraction of grid-cell covered by PFTs (fraction with bare soil + vegetation)
    !  veget_max(:,nvm) = PFT fractions of vegtot+frac_nobio
    !  veget(:,nvm) =  fractions (of vegtot+frac_nobio) covered by vegetation in each PFT
    !       BUT veget(:,1)=veget_max(:,1)
    !  frac_bare(:,nvm) = fraction (of veget_max) with bare soil in each PFT
    !  tot_bare_soil(:) = fraction of grid mesh covered by all bare soil (=SUM(frac_bare*veget_max))
    !  frac_bare_ns(:,nstm) = evaporating bare soil fraction (of vegtot) per soiltile (defined in hydrol_vegupd)

    !! 1.1 Throughfall
    ! Transformation from precisol (flux from PFT jv in m2 of grid-mesh)
    ! to  precisol_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    precisol_ns(:, :) = zero
    DO jv = 1, nvm
      DO ji = 1, kjpindex
        jst = pref_soil_veg(jv)
        IF ((veget_max(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
          precisol_ns(ji, jst) = precisol_ns(ji, jst) + precisol(ji, jv) / (soiltile(ji, jst) * vegtot(ji))
        END IF
      END DO
    END DO


    !! 1.2 Bare soil evaporation and ae_ns
    ae_ns(:, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        IF (evap_bare_lim(ji) .GT. min_sechiba) THEN
          ae_ns(ji, jst) = vevapnu(ji) * evap_bare_lim_ns(ji, jst) / evap_bare_lim(ji)
        END IF
      END DO
    END DO

    !! 1.3 transpiration
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to tr_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    ! To do next: simplify the use of humrelv(ji,jv,jst) /humrel(ji,jv), since both are equal
    tr_ns(:, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      DO ji = 1, kjpindex
        IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
          tr_ns(ji, jst) = tr_ns(ji, jst) + transpir(ji, jv) * (humrelv(ji, jv, jst) / humrel(ji, jv)) / (soiltile(ji, jst) * &
&vegtot(ji))

        END IF
      END DO
    END DO

    !! 1.4 root sink
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to root_sink (flux from contributing PFTs and soil layer with another unit, in m2 of soiltile)
    rootsink(:, :, :) = zero

      IF (ok_hydrol_arch) THEN

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        ! OBS jst = 1,nstm
          DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF (humrel(ji, jv) .GT. min_sechiba .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
              IF (is_tuzet_hydrol_arch) THEN
                rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (F_absorption(ji, jv) * e_frac(ji, jv, jsl, jst) * dt_sechiba * &
&kilo_to_unit)
              ELSE
                rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (transpir(ji, jv) * e_frac(ji, jv, jsl, jst)) / (soiltile(ji, &
&jst) * vegtot(ji))
              END IF
            END IF
          END DO
        END DO
      END DO

    ELSE

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + transpir(ji, jv) * (us(ji, jv, jst, jsl) / humrel(ji, jv)) / &
&(soiltile(ji, jst) * vegtot(ji))
              ! rootsink(ji,1,jst)=0 as us(ji,jv,jst,1)=0
            END IF
          END DO
        END DO
      END DO

    END IF
    ! ok_hydrol_arch

      !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes (grid-cell average)

      IF (check_cwrr) THEN

      error = .FALSE.

      !! 2.1 precisol

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + precisol_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

      tmp_check2(:) = zero
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          tmp_check2(ji) = tmp_check2(ji) + precisol(ji, jv)
        END DO
      END DO

        DO ji = 1, kjpindex
        IF (ABS(tmp_check1(ji) - tmp_check2(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'PRECISOL SPLIT FALSE:ji=', ji, tmp_check1(ji), tmp_check2(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - tmp_check2(ji))
          WRITE(numout, *) 'vegtot', vegtot(ji)
          DO jv = 1, nvm
            WRITE(numout, '(a,i2.2,"|",F13.4,"|",F13.4,"|",3(F9.6))') 'jv,veget_max, precisol, vegetmax_soil ', jv, veget_max(ji, &
&jv), precisol(ji, jv), vegetmax_soil(ji, jv, :)
          END DO
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,precisol_ns', jst, precisol_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', &
&'PRECISOL SPLIT FALSE')
        END IF
      END DO

      !! 2.2 ae_ns and evapnu

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + ae_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

        DO ji = 1, kjpindex

          IF (ABS(tmp_check1(ji) - vevapnu(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'VEVAPNU SPLIT FALSE:ji, Sum(ae_ns), vevapnu =', ji, tmp_check1(ji), vevapnu(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - vevapnu(ji))
          WRITE(numout, *) 'ae_ns', ae_ns(ji, :)
          WRITE(numout, *) 'vegtot', vegtot(ji)
          WRITE(numout, *) 'evap_bare_lim, evap_bare_lim_ns', evap_bare_lim(ji), evap_bare_lim_ns(ji, :)
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,ae_ns', jst, ae_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'VEVAPNU SPLIT FALSE')
        END IF
      END DO

      !! 2.3 transpiration

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + tr_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

      tmp_check2(:) = zero
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          tmp_check2(ji) = tmp_check2(ji) + transpir(ji, jv)
        END DO
      END DO

        DO ji = 1, kjpindex
        IF (ABS(tmp_check1(ji) - tmp_check2(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'TRANSPIR SPLIT FALSE:ji=', ji, tmp_check1(ji), tmp_check2(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - tmp_check2(ji))
          WRITE(numout, *) 'vegtot', vegtot(ji)
          DO jv = 1, nvm
            WRITE(numout, *) 'jv,veget_max, transpir', jv, veget_max(ji, jv), transpir(ji, jv)
            DO jst = 1, nstm
              WRITE(numout, *) 'vegetmax_soil:ji,jv,jst', ji, jv, jst, vegetmax_soil(ji, jv, jst)
            END DO
          END DO
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,tr_ns', jst, tr_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', &
&'TRANSPIR SPLIT FALSE')
        END IF

      END DO

      !! 2.4 root sink

      tmp_check3(:, :) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            tmp_check3(ji, jst) = tmp_check3(ji, jst) + rootsink(ji, jsl, jst)
          END DO
        END DO
      END DO

        DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (ABS(tmp_check3(ji, jst) - tr_ns(ji, jst)) .GT. allowed_err) THEN
            WRITE(numout, *) 'ROOTSINK SPLIT FALSE:ji,jst=', ji, jst, tmp_check3(ji, jst), tr_ns(ji, jst)
            WRITE(numout, *) 'err', ABS(tmp_check3(ji, jst) - tr_ns(ji, jst))
            WRITE(numout, *) 'HUMREL(jv=1:13)', humrel(ji, :)
            WRITE(numout, *) 'TRANSPIR', transpir(ji, :)
            DO jv = 1, nvm
              WRITE(numout, *) 'jv=', jv, 'us=', us(ji, jv, jst, :)
            END DO
            error = .TRUE.
            CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', &
&'ROOTSINK SPLIT FALSE')
          END IF
        END DO
      END DO


        !! Exit if error was found previously in this subroutine
        IF (error) THEN
        WRITE(numout, *) 'One or more errors have been detected in hydrol_split_soil. Model stops.'
        CALL ipslerr_p(3, 'hydrol_split_soil', 'We will STOP now.', 'One or several fatal errors were found previously.', '')
      END IF

    END IF
    ! end of check_cwrr


  END SUBROUTINE hydrol_split_soil


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_tridiag
  !!
  !>\BRIEF        This subroutine solves a set of linear equations which has a tridiagonal coefficient matrix.
  !!
  !! DESCRIPTION  : It is only applied in the grid-cells where resolv(ji)=TRUE
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S) : mcl (global module variable)
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_tridiag

  SUBROUTINE hydrol_soil_tridiag_acc(ji, kjpindex, ins)
    !$ACC ROUTINE SEQ

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! number of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: bet
    REAL(KIND = r_std), DIMENSION(nslm) :: gam

    !_
    !& 
!& ================================================================================================================================

    IF (resolv(ji)) THEN
      bet = tmat(ji, 1, 2)
      mcl(ji, 1, ins) = rhs(ji, 1) / bet
    END IF

      DO jsl = 2, nslm

        IF (resolv(ji)) THEN

        gam(jsl) = tmat(ji, jsl - 1, 3) / bet
        bet = tmat(ji, jsl, 2) - tmat(ji, jsl, 1) * gam(jsl)
        mcl(ji, jsl, ins) = (rhs(ji, jsl) - tmat(ji, jsl, 1) * mcl(ji, jsl - 1, ins)) / bet
      END IF

    END DO

      IF (resolv(ji)) THEN
      DO jsl = nslm - 1, 1, - 1
        mcl(ji, jsl, ins) = mcl(ji, jsl, ins) - gam(jsl + 1) * mcl(ji, jsl + 1, ins)
      END DO
    END IF

  END SUBROUTINE hydrol_soil_tridiag_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_tridiag
    !!
    !>\BRIEF        This subroutine solves a set of linear equations which has a tridiagonal coefficient matrix.
    !!
    !! DESCRIPTION  : It is only applied in the grid-cells where resolv(ji)=TRUE
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S) : mcl (global module variable)
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_tridiag

    SUBROUTINE hydrol_soil_tridiag(kjpindex, ins)

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! number of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: bet
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: gam

    !_
    !& 
!& ================================================================================================================================
    DO ji = 1, kjpindex

        IF (resolv(ji)) THEN
        bet(ji) = tmat(ji, 1, 2)
        mcl(ji, 1, ins) = rhs(ji, 1) / bet(ji)
      END IF
    END DO

      DO jsl = 2, nslm
      DO ji = 1, kjpindex

          IF (resolv(ji)) THEN

          gam(ji, jsl) = tmat(ji, jsl - 1, 3) / bet(ji)
          bet(ji) = tmat(ji, jsl, 2) - tmat(ji, jsl, 1) * gam(ji, jsl)
          mcl(ji, jsl, ins) = (rhs(ji, jsl) - tmat(ji, jsl, 1) * mcl(ji, jsl - 1, ins)) / bet(ji)
        END IF

      END DO
    END DO

      DO ji = 1, kjpindex
      IF (resolv(ji)) THEN
        DO jsl = nslm - 1, 1, - 1
          mcl(ji, jsl, ins) = mcl(ji, jsl, ins) - gam(ji, jsl + 1) * mcl(ji, jsl + 1, ins)
        END DO
      END IF
    END DO

  END SUBROUTINE hydrol_soil_tridiag

  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_nudge_mc
  !!
  !>\BRIEF         Applay nuding for soil moisture
  !!
  !! DESCRIPTION  : Applay nudging for soil moisture. The nuding values were previously read and interpolated using
  !!                the subroutine hydrol_nudge_mc_read
  !!                This subroutine is called from a loop over all soil tiles.
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! \n
  !_
  !& 
!& ================================================================================================================================
  SUBROUTINE hydrol_nudge_mc_acc(ji, kjpindex, jst, mc_loc)
    !$ACC ROUTINE SEQ

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: jst
    !! Index for current soil tile

    !! 0.2 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(INOUT) :: mc_loc
    !! Soil moisture

    !! 0.2 Locals variables
    REAL(KIND = r_std), DIMENSION(nslm, nstm) :: mc_aux
    !! Temorary variable for calculation of nudgincsm
    INTEGER(KIND = i_std) :: jsl
    !! loop index


    !! 1.5 Applay nudging of soil moisture using alpha_nudge_mc at each model sechiba time step.
    !!     alpha_mc_nudge calculated using the parameter for relaxation time NUDGE_TAU_MC set in module constantes.
    !!     alpha_nudge_mc is between 0-1
    !!     If alpha_nudge_mc=1, the new mc will be replaced by the one read from file
    mc_loc(ji, :, jst) = (1 - alpha_nudge_mc) * mc_loc(ji, :, jst) + alpha_nudge_mc * mc_read_current(ji, :, jst)


    !! 1.6 Calculate diagnostic for nudging increment of water in soil moisture
    !!     Here calculate tmc_aux for the current soil tile. Later in hydrol_nudge_mc_diag, this will be used to calculate nudgincsm
    mc_aux(:, jst) = alpha_nudge_mc * (mc_read_current(ji, :, jst) - mc_loc(ji, :, jst))
    tmc_aux(ji, jst) = dz(2) * (trois * mc_aux(1, jst) + mc_aux(2, jst)) / huit
    DO jsl = 2, nslm - 1
      tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(jsl) * (trois * mc_aux(jsl, jst) + mc_aux(jsl - 1, jst)) / huit + dz(jsl + 1) * &
&(trois * mc_aux(jsl, jst) + mc_aux(jsl + 1, jst)) / huit
    END DO
    tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(nslm) * (trois * mc_aux(nslm, jst) + mc_aux(nslm - 1, jst)) / huit


  END SUBROUTINE hydrol_nudge_mc_acc

    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_nudge_mc
    !!
    !>\BRIEF         Applay nuding for soil moisture
    !!
    !! DESCRIPTION  : Applay nudging for soil moisture. The nuding values were previously read and interpolated using
    !!                the subroutine hydrol_nudge_mc_read
    !!                This subroutine is called from a loop over all soil tiles.
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! \n
    !_
    !& 
!& ================================================================================================================================
    SUBROUTINE hydrol_nudge_mc(kjpindex, jst, mc_loc)

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: jst
    !! Index for current soil tile

    !! 0.2 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(INOUT) :: mc_loc
    !! Soil moisture

    !! 0.2 Locals variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mc_aux
    !! Temorary variable for calculation of nudgincsm
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    !! loop index


    !! 1.5 Applay nudging of soil moisture using alpha_nudge_mc at each model sechiba time step.
    !!     alpha_mc_nudge calculated using the parameter for relaxation time NUDGE_TAU_MC set in module constantes.
    !!     alpha_nudge_mc is between 0-1
    !!     If alpha_nudge_mc=1, the new mc will be replaced by the one read from file
    mc_loc(:, :, jst) = (1 - alpha_nudge_mc) * mc_loc(:, :, jst) + alpha_nudge_mc * mc_read_current(:, :, jst)


    !! 1.6 Calculate diagnostic for nudging increment of water in soil moisture
    !!     Here calculate tmc_aux for the current soil tile. Later in hydrol_nudge_mc_diag, this will be used to calculate nudgincsm
    mc_aux(:, :, jst) = alpha_nudge_mc * (mc_read_current(:, :, jst) - mc_loc(:, :, jst))
    DO ji = 1, kjpindex
      tmc_aux(ji, jst) = dz(2) * (trois * mc_aux(ji, 1, jst) + mc_aux(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(jsl) * (trois * mc_aux(ji, jsl, jst) + mc_aux(ji, jsl - 1, jst)) / huit + dz(jsl &
&+ 1) * (trois * mc_aux(ji, jsl, jst) + mc_aux(ji, jsl + 1, jst)) / huit
      END DO
      tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(nslm) * (trois * mc_aux(ji, nslm, jst) + mc_aux(ji, nslm - 1, jst)) / huit
    END DO


  END SUBROUTINE hydrol_nudge_mc


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_diag_soil
  !!
  !>\BRIEF        Calculates diagnostic variables at the grid-cell scale
  !!
  !! DESCRIPTION  :
  !! - 1. Apply mask_soiltile
  !! - 2. Sum 3d variables in 2d variables with fraction of vegetation per soil type
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne for the claculation of shumdiag_perma
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_diag_soil

  SUBROUTINE hydrol_diag_soil_acc(ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, &
&evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, &
&drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
    !$ACC ROUTINE SEQ
    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! input scalar
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! Max. vegetation type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: returnflow
    !! Water returning to the deep reservoir
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: reinfiltration
    !! Water returning to the top of the soil
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: irrigation
    !! Water from irrigation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_melt
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rain precipitation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: totfrac_nobio
    !! Total fraction of continental ice+lakes+...
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area

    !! 0.2 Output variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drysoil_frac
    !! Function of litter wetness
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff
    !! complete runoff
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drainage
    !! Drainage
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag
    !! relative soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag_perma
    !! Percent of porosity filled with water (mc/mcs) used for the thermal computations
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: k_litt
    !! litter cond.
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: litterhumdiag
    !! litter humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: vegstress
    !! Veg. moisture stress (only for vegetation growth)

    !! 0.3 Modified variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: vevapnu
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)


    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    REAL(KIND = r_std) :: mask_vegtot
    REAL(KIND = r_std) :: tmc_litter_ratio
    REAL(KIND = r_std) :: k_tmp

    !_
    !& 
!& ================================================================================================================================
    !
    ! Put the prognostics variables of soil to zero if soiltype is zero

    !! 1. Apply mask_soiltile

    DO jst = 1, nstm

      ae_ns(ji, jst) = ae_ns(ji, jst) * mask_soiltile(ji, jst)
      dr_ns(ji, jst) = dr_ns(ji, jst) * mask_soiltile(ji, jst)
      ru_ns(ji, jst) = ru_ns(ji, jst) * mask_soiltile(ji, jst)
      tmc(ji, jst) = tmc(ji, jst) * mask_soiltile(ji, jst)

        DO jv = 1, nvm
        humrelv(ji, jv, jst) = humrelv(ji, jv, jst) * mask_soiltile(ji, jst)
        DO jsl = 1, nslm
          us(ji, jv, jst, jsl) = us(ji, jv, jst, jsl) * mask_soiltile(ji, jst)
        END DO
      END DO

        DO jsl = 1, nslm
        mc(ji, jsl, jst) = mc(ji, jsl, jst) * mask_soiltile(ji, jst)
      END DO

    END DO

    runoff(ji) = zero
    drainage(ji) = zero
    humtot(ji) = zero
    shumdiag(ji, :) = zero
    shumdiag_perma(ji, :) = zero
    k_litt(ji) = zero
    litterhumdiag(ji) = zero
    tmc_litt_dry_mea(ji) = zero
    tmc_litt_wet_mea(ji) = zero
    tmc_litt_mea(ji) = zero
    humrel(ji, :) = zero
    vegstress(ji, :) = zero
    IF (ok_freeze_cwrr) THEN
      profil_froz_hydro(ji, :) = zero
      ! initialisation for the mean of profil_froz_hydro_ns
    END IF

    !! 2. Sum 3d variables in 2d variables with fraction of vegetation per soil type

    mask_vegtot = 0
    IF (vegtot(ji) .GT. min_sechiba) THEN
      mask_vegtot = 1
    END IF

    ! Here we weight ae_ns by the fraction of bare evaporating soil.
    ! This is given by frac_bare_ns, taking into account bare soil under vegetation
    ae_ns(ji, :) = mask_vegtot * ae_ns(ji, :) * frac_bare_ns(ji, :)

      ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
      DO jst = 1, nstm
      drainage(ji) = mask_vegtot * (drainage(ji) + vegtot(ji) * soiltile(ji, jst) * dr_ns(ji, jst))
      runoff(ji) = mask_vegtot * (runoff(ji) + vegtot(ji) * soiltile(ji, jst) * ru_ns(ji, jst)) + (1 - mask_vegtot) * &
&(tot_melt(ji) + irrigation(ji) + returnflow(ji) + reinfiltration(ji))
      humtot(ji) = mask_vegtot * (humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst))
      IF (ok_freeze_cwrr) THEN
        !  profil_froz_hydro_ns comes from hydrol_soil, to remain the same as in the prognotic loop
        profil_froz_hydro(ji, :) = mask_vegtot * (profil_froz_hydro(ji, :) + vegtot(ji) * soiltile(ji, jst) * &
&profil_froz_hydro_ns(ji, :, jst))
      END IF
    END DO

      ! we add the excess of snow sublimation to vevapnu
      ! - because vevapsno is modified in hydrol_snow if subsinksoil
      ! - it is multiplied by vegtot because it is devided by 1-tot_frac_nobio at creation in hydrol_snow

      IF (vegtot(ji) .NE. 0.) THEN
      vevapnu(ji) = vevapnu(ji) + subsinksoil(ji) * vegtot(ji)
    ELSE
      vevapnu(ji) = vevapnu(ji) + subsinksoil(ji)
    END IF
    runoff(ji) = runoff(ji) + precip_rain(ji) * totfrac_nobio(ji) * (1 - frac_snow_nobio(ji, iice))

      DO jst = 1, nstm
      DO jv = 1, nvm
        IF (veget_max(ji, jv) .GT. min_sechiba) THEN
          vegstress(ji, jv) = vegstress(ji, jv) + vegstressv(ji, jv, jst)
          vegstress(ji, jv) = MAX(vegstress(ji, jv), zero)
        END IF
      END DO
    END DO

      DO jst = 1, nstm
      DO jv = 1, nvm
        humrel(ji, jv) = humrel(ji, jv) + humrelv(ji, jv, jst)
        humrel(ji, jv) = MAX(humrel(ji, jv), zero)
      END DO
    END DO

      !! Litter... the goal is to calculate drysoil_frac, to calculate the albedo in condveg
      ! In condveg, drysoil_frac serve to calculate the albedo of drysoil, excluding the nobio contribution which is further added
      ! In conclusion, we calculate drysoil_frac based on moisture averages restricted to the soiltile (no multiplication by vegtot)
      ! BUT THIS IS NOT USED ANYMORE WITH THE NEW BACKGROUNG ALBEDO
      !! k_litt is calculated here as a grid-cell average (for consistency with drainage)
      !! litterhumdiag, like shumdiag, is averaged over the soiltiles for transmission to stomate
      DO jst = 1, nstm
      ! We compute here a mean k for the 'litter' used for reinfiltration from floodplains of ponds
        IF (tmc_litter(ji, jst) < tmc_litter_res(ji, jst)) THEN
        i = imin
      ELSE
        tmc_litter_ratio = (tmc_litter(ji, jst) - tmc_litter_res(ji, jst)) / (tmc_litter_sat(ji, jst) - tmc_litter_res(ji, jst))
        i = MAX(MIN(INT((imax - imin) * tmc_litter_ratio) + imin, imax - 1), imin)
      END IF
      k_tmp = MAX(k_lin(i, 1, ji) * ks(ji), zero)
      k_litt(ji) = k_litt(ji) + vegtot(ji) * soiltile(ji, jst) * SQRT(k_tmp)
      ! grid-cell average
      litterhumdiag(ji) = litterhumdiag(ji) + soil_wet_litter(ji, jst) * soiltile(ji, jst)

      tmc_litt_wet_mea(ji) = tmc_litt_wet_mea(ji) + tmc_litter_awet(ji, jst) * soiltile(ji, jst)

      tmc_litt_dry_mea(ji) = tmc_litt_dry_mea(ji) + tmc_litter_adry(ji, jst) * soiltile(ji, jst)

      tmc_litt_mea(ji) = tmc_litt_mea(ji) + tmc_litter(ji, jst) * soiltile(ji, jst)
    END DO

      IF (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji) > zero) THEN
      drysoil_frac(ji) = un + MAX(MIN((tmc_litt_dry_mea(ji) - tmc_litt_mea(ji)) / (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji)), &
&zero), - un)
    ELSE
      drysoil_frac(ji) = zero
    END IF

    ! Calculate soilmoist, as a function of total water content (mc)
    ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
    soilmoist(ji, :) = zero
    DO jst = 1, nstm
      soilmoist(ji, 1) = soilmoist(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        soilmoist(ji, jsl) = soilmoist(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) &
&/ huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist(ji, nslm) = soilmoist(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, &
&jst)) / huit
    END DO
    soilmoist(ji, :) = soilmoist(ji, :) * vegtot(ji)
    ! conversion to grid-cell average

    soilmoist_s(ji, :, :) = zero
    DO jst = 1, nstm
      soilmoist_s(ji, 1, nstm) = soilmoist_s(ji, 1, nstm) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) &
&/ huit
      DO jsl = 2, nslm - 1
        soilmoist_s(ji, jsl, nstm) = soilmoist_s(ji, jsl, nstm) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + &
&mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist_s(ji, nslm, nstm) = soilmoist_s(ji, nslm, nstm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + &
&mc(ji, nslm - 1, jst)) / huit
    END DO
    soilmoist_s(ji, :, :) = soilmoist_s(ji, :, :) * vegtot(ji)
    ! conversion to grid-cell average

    soilmoist_liquid(ji, :) = zero
    DO jst = 1, nstm
      soilmoist_liquid(ji, 1) = soilmoist_liquid(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) &
&/ huit
      DO jsl = 2, nslm - 1
        soilmoist_liquid(ji, jsl) = soilmoist_liquid(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mcl(ji, jsl, jst) + &
&mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist_liquid(ji, nslm) = soilmoist_liquid(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mcl(ji, nslm, jst) + &
&mcl(ji, nslm - 1, jst)) / huit
    END DO
    soilmoist_liquid(ji, :) = soilmoist_liquid(ji, :) * vegtot_old(ji)
    ! grid cell average


      ! Shumdiag: we start from soil_wet_ns, change the range over which the relative moisture is calculated,
      ! then do a spatial average, excluding the nobio fraction on which stomate doesn't act
      DO jst = 1, nstm
      DO jsl = 1, nslm
        shumdiag(ji, jsl) = shumdiag(ji, jsl) + soil_wet_ns(ji, jsl, jst) * soiltile(ji, jst) * ((mcs(ji) - mcw(ji)) / (mcfc(ji) - &
&mcw(ji)))
        shumdiag(ji, jsl) = MAX(MIN(shumdiag(ji, jsl), un), zero)
      END DO
    END DO

      ! Shumdiag_perma is based on soilmoist / moisture at saturation in the layer
      ! Her we start from grid averages by hydrol soil layer and transform it to the diag levels
      ! We keep a grid-cell average, like for all variables transmitted to ok_freeze
      DO jsl = 1, nslm
      shumdiag_perma(ji, jsl) = soilmoist(ji, jsl) / (dh(jsl) * mcs(ji))
      shumdiag_perma(ji, jsl) = MAX(MIN(shumdiag_perma(ji, jsl), un), zero)
    END DO

  END SUBROUTINE hydrol_diag_soil_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_diag_soil
    !!
    !>\BRIEF        Calculates diagnostic variables at the grid-cell scale
    !!
    !! DESCRIPTION  :
    !! - 1. Apply mask_soiltile
    !! - 2. Sum 3d variables in 2d variables with fraction of vegetation per soil type
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne for the claculation of shumdiag_perma
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_diag_soil

    SUBROUTINE hydrol_diag_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, &
&evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, &
&drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! input scalar
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! Max. vegetation type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: returnflow
    !! Water returning to the deep reservoir
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: reinfiltration
    !! Water returning to the top of the soil
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: irrigation
    !! Water from irrigation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_melt
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rain precipitation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: totfrac_nobio
    !! Total fraction of continental ice+lakes+...
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area

    !! 0.2 Output variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drysoil_frac
    !! Function of litter wetness
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff
    !! complete runoff
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drainage
    !! Drainage
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag
    !! relative soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag_perma
    !! Percent of porosity filled with water (mc/mcs) used for the thermal computations
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: k_litt
    !! litter cond.
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: litterhumdiag
    !! litter humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: vegstress
    !! Veg. moisture stress (only for vegetation growth)

    !! 0.3 Modified variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: vevapnu
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)


    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mask_vegtot
    REAL(KIND = r_std) :: tmc_litter_ratio
    REAL(KIND = r_std) :: k_tmp

    !_
    !& 
!& ================================================================================================================================
    !
    ! Put the prognostics variables of soil to zero if soiltype is zero

    !! 1. Apply mask_soiltile

    DO jst = 1, nstm
      DO ji = 1, kjpindex

        ae_ns(ji, jst) = ae_ns(ji, jst) * mask_soiltile(ji, jst)
        dr_ns(ji, jst) = dr_ns(ji, jst) * mask_soiltile(ji, jst)
        ru_ns(ji, jst) = ru_ns(ji, jst) * mask_soiltile(ji, jst)
        tmc(ji, jst) = tmc(ji, jst) * mask_soiltile(ji, jst)

          DO jv = 1, nvm
          humrelv(ji, jv, jst) = humrelv(ji, jv, jst) * mask_soiltile(ji, jst)
          DO jsl = 1, nslm
            us(ji, jv, jst, jsl) = us(ji, jv, jst, jsl) * mask_soiltile(ji, jst)
          END DO
        END DO

          DO jsl = 1, nslm
          mc(ji, jsl, jst) = mc(ji, jsl, jst) * mask_soiltile(ji, jst)
        END DO

      END DO
    END DO

    runoff(:) = zero
    drainage(:) = zero
    humtot(:) = zero
    shumdiag(:, :) = zero
    shumdiag_perma(:, :) = zero
    k_litt(:) = zero
    litterhumdiag(:) = zero
    tmc_litt_dry_mea(:) = zero
    tmc_litt_wet_mea(:) = zero
    tmc_litt_mea(:) = zero
    humrel(:, :) = zero
    vegstress(:, :) = zero
    IF (ok_freeze_cwrr) THEN
      profil_froz_hydro(:, :) = zero
      ! initialisation for the mean of profil_froz_hydro_ns
    END IF

      !! 2. Sum 3d variables in 2d variables with fraction of vegetation per soil type

      DO ji = 1, kjpindex
      mask_vegtot(ji) = 0
      IF (vegtot(ji) .GT. min_sechiba) THEN
        mask_vegtot(ji) = 1
      END IF
    END DO

      DO ji = 1, kjpindex
      ! Here we weight ae_ns by the fraction of bare evaporating soil.
      ! This is given by frac_bare_ns, taking into account bare soil under vegetation
      ae_ns(ji, :) = mask_vegtot(ji) * ae_ns(ji, :) * frac_bare_ns(ji, :)
    END DO

      ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
      DO jst = 1, nstm
      DO ji = 1, kjpindex
        drainage(ji) = mask_vegtot(ji) * (drainage(ji) + vegtot(ji) * soiltile(ji, jst) * dr_ns(ji, jst))
        runoff(ji) = mask_vegtot(ji) * (runoff(ji) + vegtot(ji) * soiltile(ji, jst) * ru_ns(ji, jst)) + (1 - mask_vegtot(ji)) * &
&(tot_melt(ji) + irrigation(ji) + returnflow(ji) + reinfiltration(ji))
        humtot(ji) = mask_vegtot(ji) * (humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst))
        IF (ok_freeze_cwrr) THEN
          !  profil_froz_hydro_ns comes from hydrol_soil, to remain the same as in the prognotic loop
          profil_froz_hydro(ji, :) = mask_vegtot(ji) * (profil_froz_hydro(ji, :) + vegtot(ji) * soiltile(ji, jst) * &
&profil_froz_hydro_ns(ji, :, jst))
        END IF
      END DO
    END DO

      ! we add the excess of snow sublimation to vevapnu
      ! - because vevapsno is modified in hydrol_snow if subsinksoil
      ! - it is multiplied by vegtot because it is devided by 1-tot_frac_nobio at creation in hydrol_snow

      DO ji = 1, kjpindex
      IF (vegtot(ji) .NE. 0.) THEN
        vevapnu(ji) = vevapnu(ji) + subsinksoil(ji) * vegtot(ji)
      ELSE
        vevapnu(ji) = vevapnu(ji) + subsinksoil(ji)
      END IF
      runoff(ji) = runoff(ji) + precip_rain(ji) * totfrac_nobio(ji) * (1 - frac_snow_nobio(ji, iice))
    END DO

      DO jst = 1, nstm
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          IF (veget_max(ji, jv) .GT. min_sechiba) THEN
            vegstress(ji, jv) = vegstress(ji, jv) + vegstressv(ji, jv, jst)
            vegstress(ji, jv) = MAX(vegstress(ji, jv), zero)
          END IF
        END DO
      END DO
    END DO

      DO jst = 1, nstm
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          humrel(ji, jv) = humrel(ji, jv) + humrelv(ji, jv, jst)
          humrel(ji, jv) = MAX(humrel(ji, jv), zero)
        END DO
      END DO
    END DO

      !! Litter... the goal is to calculate drysoil_frac, to calculate the albedo in condveg
      ! In condveg, drysoil_frac serve to calculate the albedo of drysoil, excluding the nobio contribution which is further added
      ! In conclusion, we calculate drysoil_frac based on moisture averages restricted to the soiltile (no multiplication by vegtot)
      ! BUT THIS IS NOT USED ANYMORE WITH THE NEW BACKGROUNG ALBEDO
      !! k_litt is calculated here as a grid-cell average (for consistency with drainage)
      !! litterhumdiag, like shumdiag, is averaged over the soiltiles for transmission to stomate
      DO jst = 1, nstm
      DO ji = 1, kjpindex
        ! We compute here a mean k for the 'litter' used for reinfiltration from floodplains of ponds
          IF (tmc_litter(ji, jst) < tmc_litter_res(ji, jst)) THEN
          i = imin
        ELSE
          tmc_litter_ratio = (tmc_litter(ji, jst) - tmc_litter_res(ji, jst)) / (tmc_litter_sat(ji, jst) - tmc_litter_res(ji, jst))
          i = MAX(MIN(INT((imax - imin) * tmc_litter_ratio) + imin, imax - 1), imin)
        END IF
        k_tmp = MAX(k_lin(i, 1, ji) * ks(ji), zero)
        k_litt(ji) = k_litt(ji) + vegtot(ji) * soiltile(ji, jst) * SQRT(k_tmp)
        ! grid-cell average
      END DO
      DO ji = 1, kjpindex
        litterhumdiag(ji) = litterhumdiag(ji) + soil_wet_litter(ji, jst) * soiltile(ji, jst)

        tmc_litt_wet_mea(ji) = tmc_litt_wet_mea(ji) + tmc_litter_awet(ji, jst) * soiltile(ji, jst)

        tmc_litt_dry_mea(ji) = tmc_litt_dry_mea(ji) + tmc_litter_adry(ji, jst) * soiltile(ji, jst)

        tmc_litt_mea(ji) = tmc_litt_mea(ji) + tmc_litter(ji, jst) * soiltile(ji, jst)
      END DO
    END DO

      DO ji = 1, kjpindex
      IF (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji) > zero) THEN
        drysoil_frac(ji) = un + MAX(MIN((tmc_litt_dry_mea(ji) - tmc_litt_mea(ji)) / (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji)), &
&zero), - un)
      ELSE
        drysoil_frac(ji) = zero
      END IF
    END DO

    ! Calculate soilmoist, as a function of total water content (mc)
    ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
    soilmoist(:, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        soilmoist(ji, 1) = soilmoist(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
        DO jsl = 2, nslm - 1
          soilmoist(ji, jsl) = soilmoist(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, &
&jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist(ji, nslm) = soilmoist(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, &
&jst)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      soilmoist(ji, :) = soilmoist(ji, :) * vegtot(ji)
      ! conversion to grid-cell average
    END DO

    soilmoist_s(:, :, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        soilmoist_s(ji, 1, nstm) = soilmoist_s(ji, 1, nstm) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, &
&jst)) / huit
        DO jsl = 2, nslm - 1
          soilmoist_s(ji, jsl, nstm) = soilmoist_s(ji, jsl, nstm) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + &
&mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist_s(ji, nslm, nstm) = soilmoist_s(ji, nslm, nstm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + &
&mc(ji, nslm - 1, jst)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      soilmoist_s(ji, :, :) = soilmoist_s(ji, :, :) * vegtot(ji)
      ! conversion to grid-cell average
    END DO

    soilmoist_liquid(:, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        soilmoist_liquid(ji, 1) = soilmoist_liquid(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, &
&jst)) / huit
        DO jsl = 2, nslm - 1
          soilmoist_liquid(ji, jsl) = soilmoist_liquid(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mcl(ji, jsl, jst) + &
&mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist_liquid(ji, nslm) = soilmoist_liquid(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mcl(ji, nslm, jst) + &
&mcl(ji, nslm - 1, jst)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      soilmoist_liquid(ji, :) = soilmoist_liquid(ji, :) * vegtot_old(ji)
      ! grid cell average
    END DO


      ! Shumdiag: we start from soil_wet_ns, change the range over which the relative moisture is calculated,
      ! then do a spatial average, excluding the nobio fraction on which stomate doesn't act
      DO jst = 1, nstm
      DO jsl = 1, nslm
        DO ji = 1, kjpindex
          shumdiag(ji, jsl) = shumdiag(ji, jsl) + soil_wet_ns(ji, jsl, jst) * soiltile(ji, jst) * ((mcs(ji) - mcw(ji)) / (mcfc(ji) &
&- mcw(ji)))
          shumdiag(ji, jsl) = MAX(MIN(shumdiag(ji, jsl), un), zero)
        END DO
      END DO
    END DO

      ! Shumdiag_perma is based on soilmoist / moisture at saturation in the layer
      ! Her we start from grid averages by hydrol soil layer and transform it to the diag levels
      ! We keep a grid-cell average, like for all variables transmitted to ok_freeze
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        shumdiag_perma(ji, jsl) = soilmoist(ji, jsl) / (dh(jsl) * mcs(ji))
        shumdiag_perma(ji, jsl) = MAX(MIN(shumdiag_perma(ji, jsl), un), zero)
      END DO
    END DO

  END SUBROUTINE hydrol_diag_soil


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_diag_soil_flux
  !!
  !>\BRIEF        : This subroutine diagnoses the vertical liquid water fluxes between the
  !!                different soil layers, based on each layer water budget. It also checks the
  !!                corresponding water conservation (during redistribution).
  !!
  !! DESCRIPTION  :
  !! 1. Initialize qflux_ns from the bottom, with dr_ns
  !! 2. Between layer nslm and nslm-1, by means of water budget knowing mc changes and flux at the lowest interface
  !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
  !! 4. Water balance verification: pursuing upward water budget, the flux at the surface should equal -flux_top
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne to fit hydrol_soil
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================

  SUBROUTINE hydrol_diag_soil_flux_acc(error_flag_hydrol_diag_soil_flux_1, ji, kjpindex, ins, mclint, flux_top)
    !$ACC ROUTINE SEQ
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_diag_soil_flux_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type
    REAL(KIND = r_std), DIMENSION(nslm), INTENT(IN) :: mclint
    !! mc values at the beginning of the time step
    REAL(KIND = r_std), INTENT(IN) :: flux_top
    !! Exfiltration (bare soil evaporation minus infiltration)

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables
    REAL(KIND = r_std) :: check_temp
    !! Diagnosed flux at soil surface, should equal -flux_top
    INTEGER(KIND = i_std) :: jsl

    !_
    !& 
!& ================================================================================================================================

    !- Compute the diffusion flux at every level from bottom to top (using mcl,mclint, and sink values)

    !! 1. Initialize qflux_ns from the bottom, with dr_ns
    jsl = nslm
    qflux_ns(ji, jsl, ins) = dr_ns(ji, ins)
    !! 2. Between layer nslm and nslm-1, by means of water budget
    !!    knowing mc changes and flux at the lowest interface
    !     qflux_ns is downward
    jsl = nslm - 1
    qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(jsl) + trois * mcl(ji, jsl + 1, ins) - trois &
&* mclint(jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins)

      !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
      ! Here, qflux_ns(ji,1,ins) is the downward flux between the top soil layer and the 2nd one
      DO jsl = nslm - 2, 1, - 1
      qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(jsl) + trois * mcl(ji, jsl + 1, ins) - &
&trois * mclint(jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins) + (dz(jsl + 2) / huit) * (trois * mcl(ji, jsl + 1, &
&ins) - trois * mclint(jsl + 1) + mcl(ji, jsl + 2, ins) - mclint(jsl + 2))
    END DO

    !! 4. Water balance verification: pursuing upward water budget, the flux at the surface (check_temp)
    !! should equal -flux_top

    check_temp = qflux_ns(ji, 1, ins) + (dz(2) / huit) * (trois * (mcl(ji, 1, ins) - mclint(1)) + (mcl(ji, 2, ins) - mclint(2))) + &
&rootsink(ji, 1, ins)
    ! flux_top is positive when upward, while check_temp is positive when downward
    check_top_ns(ji, ins) = flux_top + check_temp

      IF (ABS(check_top_ns(ji, ins)) / dt_sechiba .GT. min_sechiba) THEN
      ! Diagnosed (check_temp) and imposed (flux_top) differ by more than 1.e-8 mm/s
      error_flag_hydrol_diag_soil_flux_1 = error_flag_hydrol_diag_soil_flux_1 + 1
    END IF

  END SUBROUTINE hydrol_diag_soil_flux_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_diag_soil_flux
    !!
    !>\BRIEF        : This subroutine diagnoses the vertical liquid water fluxes between the
    !!                different soil layers, based on each layer water budget. It also checks the
    !!                corresponding water conservation (during redistribution).
    !!
    !! DESCRIPTION  :
    !! 1. Initialize qflux_ns from the bottom, with dr_ns
    !! 2. Between layer nslm and nslm-1, by means of water budget knowing mc changes and flux at the lowest interface
    !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
    !! 4. Water balance verification: pursuing upward water budget, the flux at the surface should equal -flux_top
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne to fit hydrol_soil
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================

    SUBROUTINE hydrol_diag_soil_flux(kjpindex, ins, mclint, flux_top)
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: mclint
    !! mc values at the beginning of the time step
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_top
    !! Exfiltration (bare soil evaporation minus infiltration)

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_temp
    !! Diagnosed flux at soil surface, should equal -flux_top
    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: jsl

    !_
    !& 
!& ================================================================================================================================

    !- Compute the diffusion flux at every level from bottom to top (using mcl,mclint, and sink values)
    DO ji = 1, kjpindex

      !! 1. Initialize qflux_ns from the bottom, with dr_ns
      jsl = nslm
      qflux_ns(ji, jsl, ins) = dr_ns(ji, ins)
      !! 2. Between layer nslm and nslm-1, by means of water budget
      !!    knowing mc changes and flux at the lowest interface
      !     qflux_ns is downward
      jsl = nslm - 1
      qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) - &
&trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins)
    END DO

      !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
      ! Here, qflux_ns(ji,1,ins) is the downward flux between the top soil layer and the 2nd one
      DO jsl = nslm - 2, 1, - 1
      DO ji = 1, kjpindex
        qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) &
&- trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins) + (dz(jsl + 2) / huit) * (trois * mcl(ji, jsl &
&+ 1, ins) - trois * mclint(ji, jsl + 1) + mcl(ji, jsl + 2, ins) - mclint(ji, jsl + 2))
      END DO
    END DO

      !! 4. Water balance verification: pursuing upward water budget, the flux at the surface (check_temp)
      !! should equal -flux_top
      DO ji = 1, kjpindex

      check_temp(ji) = qflux_ns(ji, 1, ins) + (dz(2) / huit) * (trois * (mcl(ji, 1, ins) - mclint(ji, 1)) + (mcl(ji, 2, ins) - &
&mclint(ji, 2))) + rootsink(ji, 1, ins)
      ! flux_top is positive when upward, while check_temp is positive when downward
      check_top_ns(ji, ins) = flux_top(ji) + check_temp(ji)

        IF (ABS(check_top_ns(ji, ins)) / dt_sechiba .GT. min_sechiba) THEN
        ! Diagnosed (check_temp) and imposed (flux_top) differ by more than 1.e-8 mm/s
        WRITE(numout, *) 'Problem in the water balance, qflux_ns computation, surface fluxes', flux_top(ji), check_temp(ji)
        WRITE(numout, *) 'Diagnosed and imposed fluxes differ by more than 1.e-8 mm/s: ', check_top_ns(ji, ins)
        WRITE(numout, *) 'ji', ji, 'jsl', jsl, 'ins', ins
        WRITE(numout, *) 'mclint', mclint(ji, :)
        WRITE(numout, *) 'mcl', mcl(ji, :, ins)
        WRITE(numout, *) 'rootsink', rootsink(ji, 1, ins)
        CALL ipslerr_p(1, 'hydrol_diag_soil_flux', 'NOTE:', 'Problem in the water balance, qflux_ns computation', '')
      END IF
    END DO

  END SUBROUTINE hydrol_diag_soil_flux

  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_smooth_over_mcs2
  !!
  !>\BRIEF        : Modifies the soil moisture profile to avoid over-saturation values,
  !!                by putting the excess in ru_ns
  !!                Thus, no point remain where such "excess" values remain (is_over_mcs becomes useless)
  !!
  !! DESCRIPTION  :
  !! The "excesses" over-saturation are corrected, by directly discarding the excess as rudr_corr,
  !! to be added to ru_ns or dr_nsrunoff (via rudr_corr).
  !! Therefore, there is no more smoothing, and this helps preventing the saturation of too many layers,
  !! which leads to numerical errors with tridiag.
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. In case of over-saturation, we directly eliminate the excess via rudr_corr
  !!    The calculation of the adjustement flux needs to account for nodes n-1 and n+1.
  !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
  !!    and export the difference with the flux
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_smooth_over_mcs2

  SUBROUTINE hydrol_soil_smooth_over_mcs2_acc(ji, mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)
    !$ACC ROUTINE SEQ

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, INTENT(OUT) :: is_over_mcs
    !! Flag diagnosing over saturated soil moisture
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(nstm), INTENT(OUT) :: rudr_corr
    !! Surface runoff produced to correct excess (mm/dtstep)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std), DIMENSION(nslm) :: excess
    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine

    !_
    !& 
!& ================================================================================================================================
    !-

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

      !! 2. In case of over-saturation, we don't do any smoothing,
      !! but directly eliminate the excess as runoff (via rudr_corr)
      !    we correct the calculation of the adjustement flux, which needs to account for nodes n-1 and n+1
      !    for the calculation to remain simple and accurate, we directly drain all the oversaturated mc,
      !    without transfering to lower layers

      !! 2.1 thresholding from top to bottom, with excess defined along jsl
      DO jsl = 1, nslm
      excess(jsl) = MAX(mc(ji, jsl, ins) - mcs(ji), zero)
      ! >=0
      mc(ji, jsl, ins) = mc(ji, jsl, ins) - excess(jsl)
      ! here mc either does not change or decreases
    END DO

    !! 2.2 To ensure conservation, this needs to be balanced by additional drainage (in kg/m2/dt)
    rudr_corr(ins) = dz(2) * (trois * excess(1) + excess(2)) / huit
    ! top layer = initialisation
      DO jsl = 2, nslm - 1
      ! intermediate layers
      rudr_corr(ins) = rudr_corr(ins) + dz(jsl) * (trois * excess(jsl) + excess(jsl - 1)) / huit + dz(jsl + 1) * (trois * &
&excess(jsl) + excess(jsl + 1)) / huit
    END DO
    rudr_corr(ins) = rudr_corr(ins) + dz(nslm) * (trois * excess(nslm) + excess(nslm - 1)) / huit
    ! bottom layer
    is_over_mcs = .FALSE.

      !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux

      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) &
&+ mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci-rudr_corr
      check(ins) = tmcf - (tmci - rudr_corr(ins))
    END IF

  END SUBROUTINE hydrol_soil_smooth_over_mcs2_acc

    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_smooth_over_mcs2
    !!
    !>\BRIEF        : Modifies the soil moisture profile to avoid over-saturation values,
    !!                by putting the excess in ru_ns
    !!                Thus, no point remain where such "excess" values remain (is_over_mcs becomes useless)
    !!
    !! DESCRIPTION  :
    !! The "excesses" over-saturation are corrected, by directly discarding the excess as rudr_corr,
    !! to be added to ru_ns or dr_nsrunoff (via rudr_corr).
    !! Therefore, there is no more smoothing, and this helps preventing the saturation of too many layers,
    !! which leads to numerical errors with tridiag.
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. In case of over-saturation, we directly eliminate the excess via rudr_corr
    !!    The calculation of the adjustement flux needs to account for nodes n-1 and n+1.
    !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
    !!    and export the difference with the flux
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_smooth_over_mcs2

    SUBROUTINE hydrol_soil_smooth_over_mcs2(mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex), INTENT(OUT) :: is_over_mcs
    !! Flag diagnosing over saturated soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: rudr_corr
    !! Surface runoff produced to correct excess (mm/dtstep)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: excess
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine

    !_
    !& 
!& ================================================================================================================================
    !-

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. In case of over-saturation, we don't do any smoothing,
      !! but directly eliminate the excess as runoff (via rudr_corr)
      !    we correct the calculation of the adjustement flux, which needs to account for nodes n-1 and n+1
      !    for the calculation to remain simple and accurate, we directly drain all the oversaturated mc,
      !    without transfering to lower layers

      !! 2.1 thresholding from top to bottom, with excess defined along jsl
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        excess(ji, jsl) = MAX(mc(ji, jsl, ins) - mcs(ji), zero)
        ! >=0
        mc(ji, jsl, ins) = mc(ji, jsl, ins) - excess(ji, jsl)
        ! here mc either does not change or decreases
      END DO
    END DO

      !! 2.2 To ensure conservation, this needs to be balanced by additional drainage (in kg/m2/dt)
      DO ji = 1, kjpindex
      rudr_corr(ji, ins) = dz(2) * (trois * excess(ji, 1) + excess(ji, 2)) / huit
      ! top layer = initialisation
    END DO
    DO jsl = 2, nslm - 1
      ! intermediate layers
        DO ji = 1, kjpindex
        rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(jsl) * (trois * excess(ji, jsl) + excess(ji, jsl - 1)) / huit + dz(jsl + 1) * &
&(trois * excess(ji, jsl) + excess(ji, jsl + 1)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(nslm) * (trois * excess(ji, nslm) + excess(ji, nslm - 1)) / huit
      ! bottom layer
      is_over_mcs(ji) = .FALSE.
    END DO

      !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux

      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, &
&ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci-rudr_corr
      check(:, ins) = tmcf(:) - (tmci(:) - rudr_corr(:, ins))
    END IF

  END SUBROUTINE hydrol_soil_smooth_over_mcs2


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil
  !!
  !>\BRIEF        This routine computes soil processes with CWRR scheme (Richards equation solved by finite differences).
  !! Note that the water fluxes are in kg/m2/dt_sechiba.
  !!
  !! DESCRIPTION  :
  !! 0. Initialisation, and split 2d variables to 3d variables, per soil tile
  !! -- START MAIN LOOP (prognostic loop to update mc and mcl) OVER SOILTILES
  !! 1. FIRSTLY, WE CHANGE MC BASED ON EXTERNAL FLUXES, ALL APPLIED AT THE SOIL SURFACE
  !! 1.1 Reduces water2infilt and water2extract to their difference
  !! 1.2 To remove water2extract (including bare soilevaporation) from top layer
  !! 1.3 Infiltration
  !! 1.4 Reinfiltration of surface runoff : compute temporary surface water and extract from runoff
  !! 2. SECONDLY, WE UPDATE MC FROM DIFFUSION, INCLUDING DRAINAGE AND ROOTSINK
  !!    This will act on mcl (liquid water content) only
  !! 2.1 K and D are recomputed after infiltration
  !! 2.2 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
  !! 2.3 We define mcl (liquid water content) based on mc and profil_froz_hydro_ns
  !! 2.4 We calculate the total SM at the beginning of the routine tridiag for water conservation check
  !! 2.5 Defining where diffusion is solved : everywhere
  !! 2.6 We define the system of linear equations for mcl redistribution
  !! 2.7 Solves diffusion equations
  !! 2.8 Computes drainage = bottom boundary condition, consistent with rhs(ji,jsl=nslm)
  !! 2.9 For water conservation check during redistribution, we calculate the total liquid SM
  !!     at the end of the routine tridiag, and we compare the difference with the flux...
  !! 3. AFTER DIFFUSION/REDISTRIBUTION
  !! 3.1 Updating mc, as all the following checks against saturation will compare mc to mcs
  !! 3.2 Correct here the possible over-saturation values (subroutine hydrol_soil_smooth_over_mcs2 acts on mc)
  !!     Here hydrol_soil_smooth_over_mcs2 discard all excess as ru_corr_ns, oriented to either ru_ns or dr_ns
  !! 3.3 Negative runoff is reported to drainage
  !! 3.4 Optional block to force saturation below zwt_force
  !! 3.5 Diagnosing the effective water table depth
  !! 3.6 Diagnose under_mcr to adapt water stress calculation below
  !! 4. At the end of the prognostic calculations, we recompute important moisture variables
  !! 4.1 Total soil moisture content (water2infilt added below)
  !! 4.2 mcl is a module variable; we update it here for calculating bare soil evaporation,
  !! 5. Optional check of the water balance of soil column (if check_cwrr)
  !! 5.1 Computation of the vertical water fluxes
  !! 6. SM DIAGNOSTICS FOR OTHER ROUTINES, MODULES, OR NEXT STEP
  !! 6.1 Total soil moisture, soil moisture at litter levels, soil wetness, us, humrelv, vesgtressv
  !! 6.2 We need to turn off evaporation when is_under_mcr
  !! 6.3 Calculate the volumetric soil moisture content (mc_layh and mcl_layh) needed in thermosoil
  !! 6.4 The hydraulic conductivities exported here are the ones used in the diffusion/redistribution
  !! -- ENDING THE MAIN LOOP ON SOILTILES
  !! 7. Summing 3d variables into 2d variables
  !! 8. XIOS export of local variables, including water conservation checks
  !! 9. COMPUTING EVAP_BARE_LIM_NS FOR NEXT TIME STEP, WITH A LOOP ON SOILTILES
  !!    The principle is to run a dummy integration of the water redistribution scheme
  !!    to check if the SM profile can sustain a potential evaporation.
  !!    If not, the dummy integration is redone from the SM profile of the end of the normal integration,
  !!    with a boundary condition leading to a very severe water limitation: mc(1)=mcr
  !! 10. evap_bar_lim is the grid-cell scale beta
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================

  SUBROUTINE hydrol_soil_acc(error_flag_hydrol_diag_soil_flux_1, error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, &
&error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, &
&error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, &
&error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12, error_flag_hydrol_soil_infilt_1, &
&error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget, &
&veget_max, soiltile, njsc, reinf_slope_soil, transpir, vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, &
&reinfiltration, irrigation, tot_melt, evap_bare_lim, evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, &
&vegstress, drysoil_frac, stempdiag, snow, snowdz, tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, &
&e_frac, ksoil, altmax, root_profile, root_depth, root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, &
&frac_snow_nobio, F_absorption)
    !$ACC ROUTINE SEQ

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_2
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_1
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_soil_infilt_1
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_12
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_11
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_10
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_9
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_8
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_7
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_6
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_5
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_4
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_3
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_2
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_1
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_diag_soil_flux_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget
    !! Fraction of vegetation type
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! Map of max vegetation types [-]
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class
    !! in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydrol_arch = .TRUE.)
    !!  @tex $(m^3 s^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: reinf_slope_soil
    !! Fraction of surface runoff that reinfiltrates per soil tile
    !!  (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: returnflow
    !! Water returning to the soil from the bottom
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: reinfiltration
    !! Water returning to the top of the soil
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: irrigation
    !! Irrigation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot
    !! Potential evaporation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot_penm
    !! Potential evaporation "Penman" (Milly's correction)
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_melt
    !! Total melt from snow and ice
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: snow
    !! Snow mass
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(IN) :: snowdz
    !! Snow depth (m)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    !!  (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: v
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: u
    !! Horizontal wind speed
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tq_cdrag
    !! Surface drag coefficient
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Fraction of water transpired supplied by individual layers (no units)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: altmax
    !! Maximul active layer thickness (m). Be careful, here active means non frozen.
    !! Not related with the active soil carbon pool.
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ncirc, nparts, nelements), INTENT(IN) :: circ_class_biomass
    !! Biomass components of the model tree
    !! within a circumference class
    !! class @tex $(g C ind^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rain precipitation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: totfrac_nobio
    !! Total fraction of continental ice+lakes+...
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area


    !! 0.2 Output variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff
    !! Surface runoff
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drainage
    !! Drainage
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: evap_bare_lim
    !! Limitation factor (beta) for bare soil evaporation
    !! on each soil column (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(INOUT) :: evap_bare_lim_ns
    !! Limitation factor (beta) for bare soil evaporation
    !! on each soil column (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag
    !! Relative soil moisture in each diag soil layer
    !! with respect to (mcfc-mcw) (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag_perma
    !! Percent of porosity filled with water (mc/mcs)
    !! in each diag soil layer (for the thermal computations)
    !! (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: k_litt
    !! Litter approximated hydraulic conductivity
    !!  @tex $(mm d^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: litterhumdiag
    !! Mean of soil_wet_litter across soil tiles
    !! (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: vegstress
    !! Veg. moisture stress (only for vegetation
    !! growth) (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drysoil_frac
    !! Function of the litter humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: mc_layh
    !! Volumetric water content (liquid + ice) for each soil layer
    !! averaged over the mesh (for thermosoil)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: mcl_layh
    !! Volumetric liquid water content for each soil layer
    !! averaged over the mesh (for thermosoil)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: mc_layh_s
    !! Volumetric soil moisture content for each layer in hydrol(liquid + ice) [m3/m3]
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: mcl_layh_s
    !! Volumetric soil moisture content for each layer in hydrol(liquid) [m3/m3]
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof), INTENT(OUT) :: root_profile
    !! Normalized root mass/length fraction in each soil layer
    !! (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ndepths), INTENT(OUT) :: root_depth
    !! Node and interface numbers at which the deepest roots
    !! occur (1 to nslm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: root_deficit
    !! water deficit to reach SM target of soil column, for irrigation demand


    !! 0.3 Modified variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: vevapnu
    !! Bare soil evaporation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(INOUT) :: humrel
    !! Relative humidity (0-1, dimensionless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: ksoil
    !! Soil conductivity (a copy of k for each soil type)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)



    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    !! Indice
    INTEGER(KIND = i_std) :: jst_kfact_root
    !! Indice for kfact_root calculation
    REAL(KIND = r_std), PARAMETER :: frac_mcs = 0.66
    !! Temporary depth
    REAL(KIND = r_std) :: temp
    !! Temporary value for fluxes
    REAL(KIND = r_std) :: tmcold
    !! Total SM at beginning of hydrol_soil (kg/m2)
    REAL(KIND = r_std) :: tmcint
    !! Ancillary total SM (kg/m2)
    REAL(KIND = r_std), DIMENSION(nslm) :: mcint
    !! To save mc values for future use
    REAL(KIND = r_std), DIMENSION(nslm) :: mclint
    !! To save mcl values for future use
    LOGICAL, DIMENSION(nstm) :: is_under_mcr
    !! Identifies under residual soil moisture points
    LOGICAL :: is_over_mcs
    !! Identifies over saturated soil moisture points
    REAL(KIND = r_std) :: diff
    REAL(KIND = r_std) :: deltahum
    !!
    LOGICAL(KIND = r_std) :: test
    !!
    REAL(KIND = r_std) :: water2extract
    !! Water flux to be extracted at the soil surface
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std) :: returnflow_soil
    !! Water from the routing back to the bottom of
    !! the soil @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std) :: reinfiltration_soil
    !! Water from the routing back to the top of the
    !! soil @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: irrigation_soil
    !! Water from irrigation returning to soil moisture per soil tile
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: flux_bottom
    !! Flux at bottom of the soil column
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: flux_top
    !! Flux at top of the soil column (for bare soil evap)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: qinfilt_ns
    !! Effective infiltration flux per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: qinfilt
    !! Effective infiltration flux
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: ru_infilt_ns
    !! Surface runoff from hydrol_soil_infilt per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: ru_infilt
    !! Surface runoff from hydrol_soil_infilt
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: ru_corr_ns
    !! Surface runoff produced to correct excess per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: ru_corr
    !! Surface runoff produced to correct excess
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: ru_corr2_ns
    !! Correction of negative surface runoff per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: ru_corr2
    !! Correction of negative surface runoff
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: dr_corr_ns
    !! Drainage produced to correct excess
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: dr_corrnum_ns
    !! Drainage produced to correct numerical errors in tridiag
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: dr_corr
    !! Drainage produced to correct excess
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std) :: dr_corrnum
    !! Drainage produced to correct numerical errors in tridiag
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: dmc
    !! Delta mc when forcing saturation (zwt_force)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: dr_force_ns
    !! Delta drainage when forcing saturation (zwt_force)
    !!  per soil tile  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: dr_force
    !! Delta drainage when forcing saturation (zwt_force)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nstm) :: wtd_ns
    !! Effective water table depth (m)
    REAL(KIND = r_std) :: wtd
    !! Mean water table depth in the grid-cell (m)

    ! For the calculation of soil_wet_ns and us/humrel/vegstress
    REAL(KIND = r_std), DIMENSION(nslm) :: sm
    !! Soil moisture of each layer (liquid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: smt
    !! Soil moisture of each layer (liquid+solid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: smw
    !! Soil moisture of each layer at wilting point
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: smf
    !! Soil moisture of each layer at field capacity
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: sms
    !! Soil moisture of each layer at saturation
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(nslm) :: sm_nostress
    !! Soil moisture of each layer at which us reaches 1
    !!  @tex $(kg m^{-2})$ @endtex
    ! For water conservation checks (in mm/dtstep unless otherwise mentioned)
    REAL(KIND = r_std), DIMENSION(nstm) :: check_infilt_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(nstm) :: check1_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(nstm) :: check_tr_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(nstm) :: check_over_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(nstm) :: check_under_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std) :: tmci
    !! Total soil moisture at beginning of routine (kg/m2)
    REAL(KIND = r_std) :: tmcf
    !! Total soil moisture at end of routine (kg/m2)
    REAL(KIND = r_std) :: diag_tr
    !! Transpiration flux
    REAL(KIND = r_std) :: check_infilt
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std) :: check1
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std) :: check_tr
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std) :: check_over
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std) :: check_under
    !! Water conservation diagnostic at routine scale
    ! For irrigation triggering
    INTEGER(KIND = i_std) :: lai_irrig_trig
    !! Number of PFT per cell with LAI> LAI_IRRIG_MIN -
    ! Diagnostic of the vertical soil water fluxes
    REAL(KIND = r_std), DIMENSION(nslm) :: qflux
    !! Local upward flux into soil layer
    !! from lower interface
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std) :: check_top
    !! Water budget residu in top soil layer
    !!  @tex $(kg m^{-2})$ @endtex

    ! Variables for calculation of a soil resistance, option do_rsoil (following the formulation of Sellers et al 1992, implemented
    !& in Oleson et al. 2008)
    REAL(KIND = r_std) :: speed
    !! magnitude of wind speed required for Aerodynamic resistance
    REAL(KIND = r_std) :: ra
    !! diagnosed aerodynamic resistance
    REAL(KIND = r_std) :: mc_rel
    !! first layer relative soil moisture, required for rsoil
    REAL(KIND = r_std) :: evap_soil
    !! soil evaporation from Oleson et al 2008
    REAL(KIND = r_std), DIMENSION(nstm) :: r_soil_ns
    !! soil resistance from Oleson et al 2008
    REAL(KIND = r_std) :: r_soil
    !! soil resistance from Oleson et al 2008
    REAL(KIND = r_std) :: tmcs_litter
    !! Saturated soil moisture in the 4 "litter" soil layers
    REAL(KIND = r_std), DIMENSION(nslm) :: root_profile_tmp
    !! Temporary variable to calculate the root_profile

    ! For CMIP6 and SP-MIP : ksat and matric pressure head psi(theta)
    REAL(KIND = r_std) :: avg
    REAL(KIND = r_std) :: mvg
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: psi
    !! Matric head (per soil layer and soil tile) [mm=kg/m2]
    REAL(KIND = r_std), DIMENSION(nslm) :: psi_moy
    !! Mean matric head per soil layer [mm=kg/m2]
    REAL(KIND = r_std), DIMENSION(nslm) :: ksat
    !! Saturated hydraulic conductivity at each node (mm/d)
    REAL(KIND = r_std), DIMENSION(nvm, nslm, nroot_prof) :: tmp
    !! temporary variable for writing the root profiles to XIOS

    !_
    !& 
!& ================================================================================================================================

    !! 0.1 Arrays with DIMENSION(kjpindex)

    returnflow_soil = zero
    reinfiltration_soil = zero
    irrigation_soil(:) = zero
    qflux_ns(ji, :, :) = zero
    mc_layh(ji, :) = zero
    ! for thermosoil
    mcl_layh(ji, :) = zero
    ! for thermosoil
    kk(ji, :, :) = zero
    kk_moy(ji, :) = zero
    undermcr(ji) = zero
    ! needs to be initialized outside from jst loop
    ksat(:) = zero
    psi_moy(:) = zero

      !! Calculate kfact_root
      IF (kfact_root_const) THEN
      kfact_root(ji, :, :) = un
    ELSE
      !! An exponential factor is used to increase ks near the surface depending on the amount of roots in the soil
      !! through a geometric average over the vegets
      !! This comes from the PhD thesis of d'Orgeval, 2006, p82; d'Orgeval et al. 2008, Eqs. 3-4
      !! (Calibrated against Hapex-Sahel measurements)
      !! Since rev 2916: veget_max/2 is used instead of veget
      kfact_root(ji, :, :) = un
      DO jsl = 1, nslm
        DO jv = 2, nvm
          jst_kfact_root = pref_soil_veg(jv)
          IF (soiltile(ji, jst_kfact_root) .GT. min_sechiba) THEN
            kfact_root(ji, jsl, jst_kfact_root) = kfact_root(ji, jsl, jst_kfact_root) * MAX((MAXVAL(ks_usda) / ks(ji)) ** (- &
&vegetmax_soil(ji, jv, jst_kfact_root) / 2 * (humcste(jv) * zz(jsl) / mille - un) / deux), un)
          END IF
        END DO
      END DO
    END IF



      IF (ok_freeze_cwrr) THEN

        ! 0.1 Calculate the temperature and fozen fraction at the hydrological levels
        ! Calculates profil_froz_hydro_ns as a function of stempdiag and mc if ok_thermodynamical_freezing
        ! These values will be kept till the end of the prognostic loop
        DO jst = 1, nstm
        CALL hydrol_soil_froz_acc(ji, nvan, avan, mcr, mcs, kjpindex, jst, njsc, stempdiag)
      END DO

    ELSE

      profil_froz_hydro_ns(ji, :, :) = zero

    END IF

    !! 0.2 Split 2d variables to 3d variables, per soil tile
    !  Here, the evaporative fluxes are distributed over the soiltiles as a function of the
    !    corresponding control factors; they are normalized to vegtot
    !  At step 7, the reverse transformation is used for the fluxes produced in hydrol_soil
    !    flux_cell(ji)=sum(flux_ns(ji,jst)*soiltile(ji,jst)*vegtot(ji))


    CALL hydrol_split_soil_acc(error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, &
&error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, &
&error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, &
&error_flag_hydrol_split_soil_12, ji, kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, &
&tot_bare_soil, us, e_frac, F_absorption)


      !! 0.3 Common variables related to routing, with all return flow applied to the soil surface
      ! The fluxes coming from the routing are uniformly splitted into the soiltiles,
      !    but are normalized to vegtot like the above fluxes:
      !            flux_ns(ji,jst)=flux_cell(ji)/vegtot(ji)
      ! It is the case for : irrigation_soil(ji) and reinfiltration_soil(ji) cf below
      ! It is also the case for subsinksoil(ji), which is divided by (1-tot_frac_nobio) at creation in hydrol_snow
      ! AD16*** The transformation in 0.2 and 0.3 is likely to induce conservation problems
      !         when tot_frac_nobio NE 0, since sum(soiltile) NE vegtot in this case
      IF (.NOT. old_irrig_scheme) THEN
      !
        IF (.NOT. irrigated_soiltile) THEN
        IF (vegtot(ji) .GT. min_sechiba) THEN
          returnflow_soil = zero
          reinfiltration_soil = (returnflow(ji) + reinfiltration(ji)) / vegtot(ji)
          IF (soiltile(ji, irrig_st) .GT. min_sechiba) THEN
            !irrigation_soil(ji, 1:2) = 0, if irrig_st = 3. Not put because Values
            !are already zero, due to initialization
            irrigation_soil(irrig_st) = irrigation(ji) / (soiltile(ji, irrig_st) * vegtot(ji))
            !Irrigation is kg/m2 of grid cell. Here, all that water is put on
            !irrig_st (irrigated soil tile), by default = 3, for the others
            !soil tiles irrigation = zero
          END IF
        END IF
      END IF
    ELSE
      !
        IF (vegtot(ji) .GT. min_sechiba) THEN
        ! returnflow_soil is assumed to enter from the bottom, but it is not possible with CWRR
        returnflow_soil = zero
        reinfiltration_soil = (returnflow(ji) + reinfiltration(ji)) / vegtot(ji)
        irrigation_soil(:) = irrigation(ji) / vegtot(ji)
        ! irrigation_soil(ji) = irrigation(ji)/vegtot(ji)
        ! Computed for all the grid cell. New way is equivalent, and coherent
        ! with irrigation_soil new dimensions (cells, soil tiles)
        ! Irrigation is kg/m2 of grid cell. For the old irrig. scheme,
        ! irrigation soil is the same for every soil tile
        ! Next lines are in tag 2.0, deleted because values are already init to zero
        ! ELSE
        ! returnflow_soil(ji) = zero
        ! reinfiltration_soil(ji) = zero
        ! irrigation_soil(ji) = zero
        ! ENDIF
      END IF
    END IF

      !! -- START MAIN LOOP (prognostic loop to update mc and mcl) OVER SOILTILES
      !!    The called subroutines work on arrays with DIMENSION(kjpindex),
      !!    recursively used for each soiltile jst

      DO jst = 1, nstm

      is_under_mcr(jst) = .FALSE.
      is_over_mcs = .FALSE.

      !! 0.4. Keep initial values for future check-up

      ! Total moisture content (including water2infilt) is saved for balance checks at the end
      ! In hydrol_tmc_update, tmc is increased by water2infilt(ji,jst), but mc is not modified !
      tmcold = tmc(ji, jst)

        ! The value of mc is kept in mcint (nstm dimension removed), in case needed for water balance checks
        DO jsl = 1, nslm
        mcint(jsl) = mask_soiltile(ji, jst) * mc(ji, jsl, jst)
      END DO
      !
      ! Initial total moisture content : tmcint does not include water2infilt, contrarily to tmcold
      tmcint = dz(2) * (trois * mcint(1) + mcint(2)) / huit
      DO jsl = 2, nslm - 1
        tmcint = tmcint + dz(jsl) * (trois * mcint(jsl) + mcint(jsl - 1)) / huit + dz(jsl + 1) * (trois * mcint(jsl) + mcint(jsl + &
&1)) / huit
      END DO
      tmcint = tmcint + dz(nslm) * (trois * mcint(nslm) + mcint(nslm - 1)) / huit

      !! 1. FIRSTLY, WE CHANGE MC BASED ON EXTERNAL FLUXES, ALL APPLIED AT THE SOIL SURFACE
      !!   Input = water2infilt(ji,jst) + irrigation_soil(ji) + reinfiltration_soil(ji) + precisol_ns(ji,jst)
      !!      - negative evaporation fluxes (MIN(ae_ns(ji,jst),zero)+ MIN(subsinksoil(ji),zero))
      !!   Output = MAX(ae_ns(ji,jst),zero) + subsinksoil(ji) = positive evaporation flux = water2extract
      ! In practice, negative subsinksoil(ji) is not possible

      !! 1.1 Reduces water2infilt and water2extract to their difference

      ! Compares water2infilt and water2extract to keep only difference
      ! Here, temp is used as a temporary variable to store the min of water to infiltrate vs evaporate
      temp = MIN(water2infilt(ji, jst) + irrigation_soil(jst) + reinfiltration_soil - MIN(ae_ns(ji, jst), zero) - &
&MIN(subsinksoil(ji), zero) + precisol_ns(ji, jst), MAX(ae_ns(ji, jst), zero) + MAX(subsinksoil(ji), zero))

      ! The water to infiltrate at the soil surface is either 0, or the difference to what has to be evaporated
      !   - the initial water2infilt (right hand side) results from qsintveg changes with vegetation updates
      !   - irrigation_soil is the input flux to the soil surface from irrigation
      !   - reinfiltration_soil is the input flux to the soil surface from routing 'including returnflow)
      !   - eventually, water2infilt holds all fluxes to the soil surface except precisol (reduced by water2extract)
      !Note that in tag 2.0, irrigation_soil(ji), changed to be coherent with new variable dimension
      water2infilt(ji, jst) = water2infilt(ji, jst) + irrigation_soil(jst) + reinfiltration_soil - MIN(ae_ns(ji, jst), zero) - &
&MIN(subsinksoil(ji), zero) + precisol_ns(ji, jst) - temp

      ! The water to evaporate from the sol surface is either the difference to what has to be infiltrated, or 0
      !   - subsinksoil is the residual from sublimation is the snowpack is not sufficient
      !   - how are the negative values of ae_ns taken into account ???
      water2extract = MAX(ae_ns(ji, jst), zero) + MAX(subsinksoil(ji), zero) - temp

      ! Here we acknowledge that subsinksoil is part of ae_ns, but ae_ns is not used further
      ae_ns(ji, jst) = ae_ns(ji, jst) + subsinksoil(ji)

      !! 1.2 To remove water2extract (including bare soil) from top layer
      flux_top = water2extract

      !! 1.3 Infiltration

      !! Definition of flux_infilt
      ! Initialise the flux to be infiltrated
      flux_infilt = water2infilt(ji, jst)

      !! K and D are computed for the profile of mc before infiltration
      !! They depend on the fraction of soil ice, given by profil_froz_hydro_ns
      CALL hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, jst, njsc)

      !! Infiltration and surface runoff are computed
      !! Infiltration stems from comparing liquid water2infilt to initial total mc (liquid+ice)
      !! The conductivity comes from hydrol_soil_coef and relates to the liquid phase only
      !  This seems consistent with ok_freeze
      CALL hydrol_soil_infilt_acc(error_flag_hydrol_soil_infilt_1, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, jst, njsc, &
&flux_infilt, stempdiag, qinfilt_ns, ru_infilt_ns, check_infilt_ns)
      ru_ns(ji, jst) = ru_infilt_ns(jst)

        !! 1.4 Reinfiltration of surface runoff : compute temporary surface water and extract from runoff
        ! Evrything here is liquid
        ! RK: water2infilt is both a volume for future reinfiltration (in mm) and a correction term for surface runoff (in
        !& mm/dt_sechiba)
        IF (.NOT. doponds) THEN
        ! this is the general case...
        water2infilt(ji, jst) = reinf_slope_soil(ji, jst) * ru_ns(ji, jst)
      ELSE
        water2infilt(ji, jst) = zero
      END IF
      !
      ru_ns(ji, jst) = ru_ns(ji, jst) - water2infilt(ji, jst)

      !! 2. SECONDLY, WE UPDATE MC FROM DIFFUSION, INCLUDING DRAINAGE AND ROOTSINK
      !!    This will act on mcl only

      !! 2.1 K and D are recomputed after infiltration
      !! They depend on the fraction of soil ice, still given by profil_froz_hydro_ns
      CALL hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, jst, njsc)

      !! 2.2 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
      !! This process will further act on mcl only, based on a, b, d from hydrol_soil_coef
      CALL hydrol_soil_setup_acc(ji, kjpindex, jst)

        !! 2.3 We define mcl (liquid water content) based on mc and profil_froz_hydro_ns
        DO jsl = 1, nslm
        mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
        ! we always have mcl<=mc
        ! if mc>mcr, then mcl>mcr; if mc=mcr,mcl=mcr; if mc<mcr, then mcl<mcr
        ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
      END DO

        ! The value of mcl is kept in mclint (nstm dimension removed), used in the flux computation after diffusion
        DO jsl = 1, nslm
        mclint(jsl) = mask_soiltile(ji, jst) * mcl(ji, jsl, jst)
      END DO

        !! 2.3bis Diagnostic of the matric potential used for redistribution by Richards/tridiag (in m)
        !  We use VG relationship giving psi as a function of mc (mcl in our case)
        !  With patches against numerical pbs when (mc_ratio - un) becomes very slightly negative (gives NaN)
        !  or if psi become too strongly negative (pbs with xios output)
        DO jsl = 1, nslm
        IF (soiltile(ji, jst) .GT. zero) THEN
          mvg = un - un / nvan_mod_tab(jsl, ji)
          avg = avan_mod_tab(jsl, ji) * 1000.
          ! to convert in m-1
          mc_ratio = MAX(10. ** (- 14 * mvg), (mcl(ji, jsl, jst) - mcr(ji)) / (mcs(ji) - mcr(ji))) ** (- un / mvg)
          psi = - MAX(zero, (mc_ratio - un)) ** (un / nvan_mod_tab(jsl, ji)) / avg
          ! in m
          psi_moy(jsl) = psi_moy(jsl) + soiltile(ji, jst) * psi
          ! average across soil tiles
        END IF
      END DO

      !! 2.4 We calculate the total SM at the beginning of the routine tridiag for water conservation check
      !  (on mcl only, since the diffusion only modifies mcl)
      tmci = dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, &
&jst) + mcl(ji, jsl + 1, jst)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mcl(ji, nslm, jst) + mcl(ji, nslm - 1, jst)) / huit

      !! 2.5 Defining where diffusion is solved : everywhere
      !! Since mc>mcs is not possible after infiltration, and we accept that mc<mcr
      !! (corrected later by shutting off all evaporative fluxes in this case)
      !  Nothing is done if resolv=F
      resolv(ji) = (mask_soiltile(ji, jst) .GT. 0)

      !! 2.6 We define the system of linear equations for mcl redistribution,
      !! based on the matrix coefficients from hydrol_soil_setup
      !! following the PhD thesis of de Rosnay (1999), p155-157
      !! The bare soil evaporation (subtracted from infiltration) is used directly as flux_top
      ! rhs for right-hand side term; fp for f'; gp for g'; ep for e'; with flux=0 !

      !- First layer
      tmat(ji, 1, 1) = zero
      tmat(ji, 1, 2) = f(ji, 1)
      tmat(ji, 1, 3) = g1(ji, 1)
      rhs(ji, 1) = fp(ji, 1) * mcl(ji, 1, jst) + gp(ji, 1) * mcl(ji, 2, jst) - flux_top - (b(ji, 1) + b(ji, 2)) / deux * &
&(dt_sechiba / one_day) - rootsink(ji, 1, jst)
      !- soil body
        DO jsl = 2, nslm - 1
        tmat(ji, jsl, 1) = e(ji, jsl)
        tmat(ji, jsl, 2) = f(ji, jsl)
        tmat(ji, jsl, 3) = g1(ji, jsl)
        rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + gp(ji, jsl) * mcl(ji, jsl + 1, jst) &
&+ (b(ji, jsl - 1) - b(ji, jsl + 1)) * (dt_sechiba / one_day) / deux - rootsink(ji, jsl, jst)
      END DO
      !- Last layer, including drainage
      jsl = nslm
      tmat(ji, jsl, 1) = e(ji, jsl)
      tmat(ji, jsl, 2) = f(ji, jsl)
      tmat(ji, jsl, 3) = zero
      rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + (b(ji, jsl - 1) + b(ji, jsl) * (un - &
&deux * free_drain_coef(ji, jst))) * (dt_sechiba / one_day) / deux - rootsink(ji, jsl, jst)
      !- Store the equations in case needed again
        DO jsl = 1, nslm
        srhs(ji, jsl) = rhs(ji, jsl)
        stmat(ji, jsl, 1) = tmat(ji, jsl, 1)
        stmat(ji, jsl, 2) = tmat(ji, jsl, 2)
        stmat(ji, jsl, 3) = tmat(ji, jsl, 3)
      END DO

      !! 2.7 Solves diffusion equations, but only in grid-cells where resolv is true, i.e. everywhere (cf 2.2)
      !!     The result is an updated mcl profile

      CALL hydrol_soil_tridiag_acc(ji, kjpindex, jst)

        !! 2.8 Computes drainage = bottom boundary condition, consistent with rhs(ji,jsl=nslm)
        ! dr_ns in mm/dt_sechiba, from k in mm/d
        ! This should be done where resolv=T, like tridiag (drainage is part of the linear system !)
        IF (resolv(ji)) THEN
        dr_ns(ji, jst) = mask_soiltile(ji, jst) * k(ji, nslm) * free_drain_coef(ji, jst) * (dt_sechiba / one_day)
      ELSE
        dr_ns(ji, jst) = zero
      END IF

      !! 2.9 For water conservation check during redistribution AND CORRECTION,
      !!     we calculate the total liquid SM at the end of the routine tridiag
      tmcf = dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, &
&jst) + mcl(ji, jsl + 1, jst)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mcl(ji, nslm, jst) + mcl(ji, nslm - 1, jst)) / huit

      !! And we compare the difference with the flux...
      ! Normally, tcmf=tmci-flux_top(ji)-transpir-dr_ns
      diag_tr = SUM(rootsink(ji, :, jst))
      ! Here, check_tr_ns holds the inaccuracy during the redistribution phase
      check_tr_ns(jst) = tmcf - (tmci - flux_top - dr_ns(ji, jst) - diag_tr)

        !! We solve here the numerical errors that happen when the soil is close to saturation
        !! and drainage very high, and which lead to negative check_tr_ns: the soil dries more
        !! than what is demanded by the fluxes, so we need to increase the fluxes.
        !! This is done by increasing the drainage.
        !! There are also instances of positive check_tr_ns, larger when the drainage is high
        !! They are similarly corrected by a decrease of dr_ns, in the limit of keeping a positive drainage.
        IF (check_tr_ns(jst) .LT. zero) THEN
        dr_corrnum_ns(jst) = - check_tr_ns(jst)
      ELSE
        dr_corrnum_ns(jst) = - MIN(dr_ns(ji, jst), check_tr_ns(jst))
      END IF
      dr_ns(ji, jst) = dr_ns(ji, jst) + dr_corrnum_ns(jst)
      ! dr_ns increases/decrease if check_tr negative/positive
        !! For water conservation check during redistribution
        IF (check_cwrr) THEN
        check_tr_ns(jst) = tmcf - (tmci - flux_top - dr_ns(ji, jst) - diag_tr)
      END IF

        !! 3. AFTER DIFFUSION/REDISTRIBUTION

        !! 3.1 Updating mc, as all the following checks against saturation will compare mc to mcs
        !      The frozen fraction is constant, so that any water flux to/from a layer changes
        !      both mcl and the ice amount. The assumption behind this is that water entering/leaving
        !      a soil layer immediately freezes/melts with the proportion profil_froz_hydro_ns/(1-profil_...)
        DO jsl = 1, nslm
        mc(ji, jsl, jst) = MAX(mcl(ji, jsl, jst), mcl(ji, jsl, jst) + profil_froz_hydro_ns(ji, jsl, jst) * (mc(ji, jsl, jst) - &
&mcr(ji)))
        ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we get mc=mcl
      END DO

      !! 3.2 Correct here the possible over-saturation values (subroutine hydrol_soil_smooth_over_mcs2 acts on mc)
      !    Oversaturation results from numerical inaccuracies and can be frequent if free_drain_coef=0
      !    Here hydrol_soil_smooth_over_mcs2 discard all excess as ru_corr_ns, oriented to either ru_ns or dr_ns
      !    The former routine hydrol_soil_smooth_over_mcs, which keeps most of the excess in the soiltile
      !    after smoothing, first downward then upward, is kept in the module but not used here
      dr_corr_ns(jst) = zero
      ru_corr_ns(jst) = zero
      CALL hydrol_soil_smooth_over_mcs2_acc(ji, mcs, kjpindex, jst, njsc, is_over_mcs, ru_corr_ns, check_over_ns)

        ! In absence of freezing, if F is large enough, the correction of oversaturation is sent to drainage
        IF ((free_drain_coef(ji, jst) .GE. 0.5) .AND. (.NOT. ok_freeze_cwrr)) THEN
        dr_corr_ns(jst) = ru_corr_ns(jst)
        ru_corr_ns(jst) = zero
      END IF
      dr_ns(ji, jst) = dr_ns(ji, jst) + dr_corr_ns(jst)
      ru_ns(ji, jst) = ru_ns(ji, jst) + ru_corr_ns(jst)

      !! 3.3 Negative runoff is reported to drainage
      !  Since we computed ru_ns directly from hydrol_soil_infilt, ru_ns should not be negative

      ru_corr2_ns(jst) = zero
      IF (ru_ns(ji, jst) .LT. zero) THEN
        dr_ns(ji, jst) = dr_ns(ji, jst) + ru_ns(ji, jst)
        ru_corr2_ns(jst) = - ru_ns(ji, jst)
        ru_ns(ji, jst) = 0.
      END IF

        !! 3.4.1 Optional nudging for soil moisture
        IF (ok_nudge_mc) THEN
        CALL hydrol_nudge_mc_acc(ji, kjpindex, jst, mc)
      END IF


        !! 3.4.2 Optional block to force saturation below zwt_force
        ! This block is not compatible with freezing; in this case, mcl must be corrected too
        ! We test if zwt_force(1,jst) <= zmaxh, to avoid steps 1 and 2 if unnecessary

        IF (zwt_force(1, jst) <= zmaxh) THEN

          !! We force the nodes below zwt_force to be saturated
          !  As above, we compare mc to mcs
          DO jsl = 1, nslm
          dmc(jsl) = zero
          IF ((zz(jsl) >= zwt_force(ji, jst) * mille)) THEN
            dmc(jsl) = mcs(ji) - mc(ji, jsl, jst)
            ! addition to reach mcs (m3/m3) = positive value
            mc(ji, jsl, jst) = mcs(ji)
          END IF
        END DO

        !! To ensure conservation, this needs to be balanced by a negative change in drainage (in kg/m2/dt)
        dr_force_ns(jst) = dz(2) * (trois * dmc(1) + dmc(2)) / huit
        ! top layer = initialization
          DO jsl = 2, nslm - 1
          ! intermediate layers
          dr_force_ns(jst) = dr_force_ns(jst) + dz(jsl) * (trois * dmc(jsl) + dmc(jsl - 1)) / huit + dz(jsl + 1) * (trois * &
&dmc(jsl) + dmc(jsl + 1)) / huit
        END DO
        dr_force_ns(jst) = dr_force_ns(jst) + dz(nslm) * (trois * dmc(nslm) + dmc(nslm - 1)) / huit
        ! bottom layer
        dr_ns(ji, jst) = dr_ns(ji, jst) - dr_force_ns(jst)
        ! dr_force_ns is positive and dr_ns must be reduced

      ELSE

        dr_force_ns(jst) = zero

      END IF

      !! 3.5 Diagnosing the effective water table depth:
      !!     Defined as as the smallest jsl value when mc(jsl) is no more at saturation (mcs), starting from the bottom
      !      If there is a part of the soil which is saturated but underlain with unsaturated nodes,
      !      this is not considered as a water table
      wtd_ns(jst) = undef_sechiba
      ! in meters
      jsl = nslm
      DO WHILE ((mc(ji, jsl, jst) .EQ. mcs(ji)) .AND. (jsl > 1))
        wtd_ns(jst) = zz(jsl) / mille
        ! in meters
        jsl = jsl - 1
      END DO

      !! 3.6 Diagnose under_mcr to adapt water stress calculation below
      !      This routine does not change tmc but decides where we should turn off ET to prevent further mc decrease
      !      Like above, the tests are made on total mc, compared to mcr
      CALL hydrol_soil_smooth_under_mcr_acc(ji, mcr, kjpindex, jst, njsc, is_under_mcr, check_under_ns)

      !! 4. At the end of the prognostic calculations, we recompute important moisture variables

      !! 4.1 Total soil moisture content (water2infilt added below)
      tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * &
&mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
      END DO
      tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit

        !! 4.2 mcl is a module variable; we update it here for calculating bare soil evaporation,
        !!     and in case we would like to export it (xios)
        DO jsl = 1, nslm
        mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
        ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
      END DO

        !! 5. Optional check of the water balance of soil column (if check_cwrr)

        IF (check_cwrr) THEN

        !! 5.1 Computation of the vertical water fluxes and water balance of the top layer
        CALL hydrol_diag_soil_flux_acc(error_flag_hydrol_diag_soil_flux_1, ji, kjpindex, jst, mclint, flux_top)

      END IF

      !! 6. SM DIAGNOSTICS FOR OTHER ROUTINES, MODULES, OR NEXT STEP
      !    Starting here, mc and mcl should not change anymore

      !! 6.1 Total soil moisture, soil moisture at litter levels, soil wetness, us, humrelv, vesgtressv
      !!     (based on mc)

      !! In output, tmc includes water2infilt(ji,jst)
      tmc(ji, jst) = tmc(ji, jst) + water2infilt(ji, jst)

      ! The litter is the 4 top levels of the soil
      ! Compute various field of soil moisture for the litter (used for stomate and for albedo)
      ! We exclude the frozen water from the calculation
      tmc_litter(ji, jst) = dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      ! sum from level 1 to 4
        DO jsl = 2, 4
        tmc_litter(ji, jst) = tmc_litter(ji, jst) + dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl &
&+ 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit
      END DO

      ! Subsequent calculation of soil_wet_litter (tmc-tmcw)/(tmcfc-tmcw)
      ! Based on liquid water content
      soil_wet_litter(ji, jst) = MIN(un, MAX(zero, (tmc_litter(ji, jst) - tmc_litter_wilt(ji, jst)) / (tmc_litter_field(ji, jst) - &
&tmc_litter_wilt(ji, jst))))

      ! Preliminary calculation of various soil moistures (for each layer, in kg/m2)
      sm(1) = dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      smt(1) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      smw(1) = dz(2) * (quatre * mcw(ji)) / huit
      smf(1) = dz(2) * (quatre * mcfc(ji)) / huit
      sms(1) = dz(2) * (quatre * mcs(ji)) / huit
      DO jsl = 2, nslm - 1
        sm(jsl) = dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, jst) &
&+ mcl(ji, jsl + 1, jst)) / huit
        smt(jsl) = dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + &
&mc(ji, jsl + 1, jst)) / huit
        smw(jsl) = dz(jsl) * (quatre * mcw(ji)) / huit + dz(jsl + 1) * (quatre * mcw(ji)) / huit
        smf(jsl) = dz(jsl) * (quatre * mcfc(ji)) / huit + dz(jsl + 1) * (quatre * mcfc(ji)) / huit
        sms(jsl) = dz(jsl) * (quatre * mcs(ji)) / huit + dz(jsl + 1) * (quatre * mcs(ji)) / huit
      END DO
      sm(nslm) = dz(nslm) * (trois * mcl(ji, nslm, jst) + mcl(ji, nslm - 1, jst)) / huit
      smt(nslm) = dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      smw(nslm) = dz(nslm) * (quatre * mcw(ji)) / huit
      smf(nslm) = dz(nslm) * (quatre * mcfc(ji)) / huit
      sms(nslm) = dz(nslm) * (quatre * mcs(ji)) / huit
      ! sm_nostress = soil moisture of each layer at which us reaches 1, here at the middle of [smw,smf]
        DO jsl = 1, nslm
        sm_nostress(jsl) = smw(jsl) + pcent(njsc(ji)) * (smf(jsl) - smw(jsl))
      END DO

      ! Saturated litter soil moisture for rsoil
      tmcs_litter = zero
      DO jsl = 1, 4
        tmcs_litter = tmcs_litter + sms(jsl)
      END DO

        ! Here we compute root zone deficit, to have an estimate of water demand in irrigated soil column (i.e. crop and grass)
        IF (jst .EQ. irrig_st) THEN
        !It computes water deficit only on the root zone, and only on the layers where
        !there is actually a deficit. If there is not deficit, it does not take into account that layer

        root_deficit(ji) = SUM(MAX(zero, beta_irrig * smf(1 : nslm_root(ji)) - sm(1 : nslm_root(ji)))) - water2infilt(ji, jst)

        root_deficit(ji) = MAX(root_deficit(ji), zero)

        !It COUNTS the number of pft with LAI > lai_irrig_min, inside the soiltile
        !It compares veget, but it is the same as they are related by a function
        lai_irrig_trig = 0

          DO jv = 1, nvm
          IF (.NOT. natural(jv)) THEN

              IF (veget(ji, jv) > veget_max(ji, jv) * (un - EXP(- lai_irrig_min * ext_coeff_vegetfrac(jv)))) THEN

              lai_irrig_trig = lai_irrig_trig + 1

            END IF

          END IF

        END DO
        !If any of the PFT inside the soil tile have LAI >  lai_irrig_min (I.E. lai_irrig_trig(ji) = 0 )
          !The root deficit is set to zero, and irrigation is not triggered

          IF (lai_irrig_trig < 1) THEN
          root_deficit(ji) = zero
        END IF

      END IF

        ! Soil wetness profiles (W-Ww)/(Ws-Ww)
        ! soil_wet_ns is the ratio of available soil moisture to max available soil moisture
        ! (ie soil moisture at saturation minus soil moisture at wilting point).
        ! soil wet is a water stress for stomate, to control C decomposition
        ! Based on liquid water content
        DO jsl = 1, nslm
        soil_wet_ns(ji, jsl, jst) = MIN(un, MAX(zero, (sm(jsl) - smw(jsl)) / (sms(jsl) - smw(jsl))))
      END DO

      ! Compute us and the new humrelv to use in sechiba (with loops on the vegetation types)
      ! This is the water stress for transpiration (diffuco) and photosynthesis (diffuco)
      ! humrel is never used in stomate
      ! Based on liquid water content

      ! -- PFT1
      humrelv(ji, 1, jst) = zero
      ! -- Top layer
        DO jv = 2, nvm
        !- Here we make the assumption that roots do not take water from the 1st layer.
        us(ji, jv, jst, 1) = zero
        humrelv(ji, jv, jst) = zero
        ! initialisation of the sum
      END DO

      ! There are two different ways of looking at a root profile in the code. It
      ! could reflect "structure" or "function". The code uses a different
      ! root profile depending on what it is used for. A structural and functional
      ! root profile are calculated below.
      CALL hydrol_root_profile_acc(error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2, ji, kjpindex, altmax, sm, &
&smw, root_profile, root_depth)

      ! Make root_dens XIOS proof. Use NAN instead of zero to obtain the correct mean
      ! value for the period that roots are present.
      tmp(:, :, :) = root_profile(ji, :, :, :)
      DO jv = 1, nvm
        DO jsl = 1, nslm
          IF (SUM(circ_class_biomass(ji, jv, :, iroot, icarbon), DIM = 1) .LT. min_stomate) THEN
            tmp(jv, jsl, istruc) = xios_default_val
            tmp(jv, jsl, ifunc) = xios_default_val
          END IF
        END DO
      END DO
      !CALL xios_orchidee_send_field("ROOT_PROF_STRUC",tmp(:,:,:,istruc))
        !CALL xios_orchidee_send_field("ROOT_PROF_FUNC",tmp(:,:,:,ifunc))

        ! Intermediate and bottom layers
        DO jsl = 2, nslm
        DO jv = 2, nvm
          ! AD16*** Although plants can only withdraw liquid water, we compute here the water stress
            ! based on mc and the corresponding thresholds mcs, pcent, or potentially mcw and mcfc
            ! This is consistent with assuming that ice is uniformly distributed within the poral space
            ! In such a case, freezing makes mcl and the "liquid" porosity smaller than the "total" values
            ! And it is the same for all the moisture thresholds, which are proportional to porosity.
            ! Since the stress is based on relative moisture, it could thus independent from the porosity
            ! at first order, thus independent from freezing.
            ! 26-07-2017: us and humrel now based on liquid soil moisture, so the stress is stronger
            IF (new_watstress) THEN
            IF ((sm(jsl) - smw(jsl)) .GT. min_sechiba) THEN
              us(ji, jv, jst, jsl) = MIN(un, MAX(zero, (EXP(- alpha_watstress * ((smf(jsl) - smw(jsl)) / (sm_nostress(jsl) - &
&smw(jsl))) * ((sm_nostress(jsl) - sm(jsl)) / (sm(jsl) - smw(jsl))))))) * root_profile(ji, jv, jsl, ifunc)
            ELSE
              us(ji, jv, jst, jsl) = 0.
            END IF
          ELSE
            us(ji, jv, jst, jsl) = MIN(un, MAX(zero, (sm(jsl) - smw(jsl)) / (sm_nostress(jsl) - smw(jsl)))) * root_profile(ji, jv, &
&jsl, ifunc)
          END IF
          humrelv(ji, jv, jst) = humrelv(ji, jv, jst) + us(ji, jv, jst, jsl)
        END DO
      END DO

      !! vegstressv is the water stress for phenology in stomate
      !! It varies linearly from zero at wilting point to 1 at field capacity
      vegstressv(ji, :, jst) = zero
      DO jv = 2, nvm
        DO jsl = 1, nslm
          vegstressv(ji, jv, jst) = vegstressv(ji, jv, jst) + MIN(un, MAX(zero, (sm(jsl) - smw(jsl)) / (smf(jsl) - smw(jsl)))) * &
&root_profile(ji, jv, jsl, ifunc)
        END DO
      END DO


        ! -- If the PFT is absent, the corresponding humrelv and vegstressv = 0
        DO jv = 2, nvm
        IF (vegetmax_soil(ji, jv, jst) .LT. min_sechiba) THEN
          humrelv(ji, jv, jst) = zero
          vegstressv(ji, jv, jst) = zero
          us(ji, jv, jst, :) = zero
        END IF
      END DO

        !! 6.2 We need to turn off evaporation when is_under_mcr
        !!     We set us, humrelv and vegstressv to zero in this case
        !!     WARNING: It's different from having locally us=0 in the soil layers(s) where mc<mcr
        !!              This part is crucial to preserve water conservation
        DO jsl = 1, nslm
        DO jv = 2, nvm
          IF (is_under_mcr(jst)) THEN
            us(ji, jv, jst, jsl) = zero
          END IF
        END DO
      END DO
      DO jv = 2, nvm
        IF (is_under_mcr(jst)) THEN
          humrelv(ji, jv, jst) = zero
        END IF
      END DO
      !rwilt and soil_wet_ns to zero in this case.
        ! They are used later for shumdiag and shumdiag_perma
        DO jsl = 1, nslm
        IF (is_under_mcr(jst)) THEN
          soil_wet_ns(ji, jsl, jst) = zero
        END IF
      END DO
      IF (is_under_mcr(jst)) THEN
        undermcr(ji) = undermcr(ji) + un
      END IF

      !! 6.3 Calculate the volumetric soil moisture content (mc_layh and mcl_layh) needed in
      !!     thermosoil for the thermal conductivity.
      !! The multiplication by vegtot creates grid-cell average values
      ! *** To be checked for consistency with the use of nobio properties in thermosoil
      mc_layh_s(ji, :, :) = mc(ji, :, :)
      mcl_layh_s(ji, :, :) = mc(ji, :, :)
      DO jsl = 1, nslm
        mc_layh(ji, jsl) = mc_layh(ji, jsl) + mc(ji, jsl, jst) * soiltile(ji, jst) * vegtot(ji)
        mcl_layh(ji, jsl) = mcl_layh(ji, jsl) + mcl(ji, jsl, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO

      !! 6.4 The hydraulic conductivities exported here are the ones used in the diffusion/redistribution
      ! (no call of hydrol_soil_coef since 2.1)
      ! We average the values of each soiltile and keep the specific value (no multiplication by vegtot)
      kk_moy(ji, :) = kk_moy(ji, :) + soiltile(ji, jst) * k(ji, :)
      kk(ji, :, jst) = k(ji, :)

        !! 6.5 We also want to export ksat at each node for CMIP6
        !  (In the output, done only once according to field_def_orchidee.xml; same averaging as for kk)
        DO jsl = 1, nslm
        ksat(jsl) = ksat(jsl) + soiltile(ji, jst) * (ks(ji) * kfact(jsl, ji) * kfact_root(ji, jsl, jst))
      END DO


    END DO
    ! end of loop on soiltile


    !! -- ENDING THE MAIN LOOP ON SOILTILES

    !! 7. Summing 3d variables into 2d variables
    CALL hydrol_diag_soil_acc(ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, &
&evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, &
&drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)

    ! Means of wtd, runoff and drainage corrections, across soiltiles
    wtd = zero
    ru_corr = zero
    ru_corr2 = zero
    dr_corr = zero
    dr_corrnum = zero
    dr_force = zero
    DO jst = 1, nstm
      wtd = wtd + soiltile(ji, jst) * wtd_ns(jst)
      ! average over vegtot only
        IF (vegtot(ji) .GT. min_sechiba) THEN
        ! to mimic hydrol_diag_soil
        ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
        ru_corr = ru_corr + vegtot(ji) * soiltile(ji, jst) * ru_corr_ns(jst)
        ru_corr2 = ru_corr2 + vegtot(ji) * soiltile(ji, jst) * ru_corr2_ns(jst)
        dr_corr = dr_corr + vegtot(ji) * soiltile(ji, jst) * dr_corr_ns(jst)
        dr_corrnum = dr_corrnum + vegtot(ji) * soiltile(ji, jst) * dr_corrnum_ns(jst)
        dr_force = dr_force - vegtot(ji) * soiltile(ji, jst) * dr_force_ns(jst)
        ! the sign is OK to get a negative drainage flux
      END IF
    END DO

    ! Means local variables, including water conservation checks
    ru_infilt = 0.
    qinfilt = 0.
    check_infilt = 0.
    check_tr = 0.
    check_over = 0.
    check_under = 0.
    qflux(:) = 0.
    check_top = 0.
    DO jst = 1, nstm
      IF (vegtot(ji) .GT. min_sechiba) THEN
        ! to mimic hydrol_diag_soil
        ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
        ru_infilt = ru_infilt + vegtot(ji) * soiltile(ji, jst) * ru_infilt_ns(jst)
        qinfilt = qinfilt + vegtot(ji) * soiltile(ji, jst) * qinfilt_ns(jst)
      END IF
    END DO

      IF (check_cwrr) THEN
      DO jst = 1, nstm
        IF (vegtot(ji) .GT. min_sechiba) THEN
          ! to mimic hydrol_diag_soil
          ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
          check_infilt = check_infilt + vegtot(ji) * soiltile(ji, jst) * check_infilt_ns(jst)
          check_tr = check_tr + vegtot(ji) * soiltile(ji, jst) * check_tr_ns(jst)
          check_over = check_over + vegtot(ji) * soiltile(ji, jst) * check_over_ns(jst)
          check_under = check_under + vegtot(ji) * soiltile(ji, jst) * check_under_ns(jst)
          !
          qflux(:) = qflux(:) + vegtot(ji) * soiltile(ji, jst) * qflux_ns(ji, :, jst)
          check_top = check_top + vegtot(ji) * soiltile(ji, jst) * check_top_ns(ji, jst)
        END IF
      END DO
    END IF

    !! 8. COMPUTING EVAP_BARE_LIM_NS FOR NEXT TIME STEP, WITH A LOOP ON SOILTILES
    !!    The principle is to run a dummy integration of the water redistribution scheme
    !!    to check if the SM profile can sustain a potential evaporation.
    !!    If not, the dummy integration is redone from the SM profile of the end of the normal integration,
    !!    with a boundary condition leading to a very severe water limitation: mc(1)=mcr

    ! evap_bare_lim = beta factor for bare soil evaporation
    evap_bare_lim(ji) = zero
    evap_bare_lim_ns(ji, :) = zero

      ! Loop on soil tiles
      DO jst = 1, nstm

        !! 8.1 Save actual mc, mcl, and tmc for restoring at the end of the time step
        !!      and calculate tmcint corresponding to mc without water2infilt
        DO jsl = 1, nslm
        mcint(jsl) = mask_soiltile(ji, jst) * mc(ji, jsl, jst)
        mclint(jsl) = mask_soiltile(ji, jst) * mcl(ji, jsl, jst)
      END DO

      temp = tmc(ji, jst)
      tmcint = temp - water2infilt(ji, jst)
      ! to estimate bare soil evap based on water budget

        !! 8.2 Since we estimate bare soile evap for the next time step, we update profil_froz_hydro and mcl
        !     (effect of mc only, the change in stempdiag is neglected)
        IF (ok_freeze_cwrr) THEN
        CALL hydrol_soil_froz_acc(ji, nvan, avan, mcr, mcs, kjpindex, jst, njsc, stempdiag)
      END IF
      DO jsl = 1, nslm
        mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
        ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
      END DO

      !! 8.3 K and D are recomputed for the updated profile of mc/mcl
      CALL hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, jst, njsc)
      !! for the hydraulic architecture we need to pass the hydraulic
      !  conductivity. We save this variable in ksoil
      ksoil(ji, :, jst) = k(ji, :)

      !! 8.4 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
      CALL hydrol_soil_setup_acc(ji, kjpindex, jst)
      resolv(ji) = (mask_soiltile(ji, jst) .GT. 0)

        !! 8.5 We define the system of linear equations, based on matrix coefficients,

        !- Impose potential evaporation as flux_top in mm/step, assuming the water is available
        ! Note that this should lead to never have evapnu>evapot_penm(ji)


        IF (vegtot(ji) .GT. min_sechiba) THEN

          ! We calculate a reduced demand, by means of a soil resistance (Sellers et al., 1992)
          ! It is based on the liquid SM only, like for us and humrel
          IF (do_rsoil) THEN
          mc_rel = tmc_litter(ji, jst) / tmcs_litter
          ! tmc_litter based on mcl
          ! based on SM in the top 4 soil layers (litter) to smooth variability
          r_soil_ns(jst) = EXP(8.206 - 4.255 * mc_rel)
        ELSE
          r_soil_ns(jst) = zero
        END IF

        ! Aerodynamic resistance
        speed = MAX(min_wind, SQRT(u(ji) * u(ji) + v(ji) * v(ji)))
        IF (speed * tq_cdrag(ji) .GT. min_sechiba) THEN
          ra = un / (speed * tq_cdrag(ji))
          evap_soil = evapot_penm(ji) / (un + r_soil_ns(jst) / ra)
        ELSE
          evap_soil = evapot_penm(ji)
        END IF

        flux_top = evap_soil * AINT(frac_bare_ns(ji, jst) + un - min_sechiba)
      ELSE

        flux_top = zero
        ! r_soil_ns needs a value to support the calculation in
        ! section "evap_bar_lim is the grid-cell scale beta"
        r_soil_ns(jst) = zero

      END IF

      ! We don't use rootsinks, because we assume there is no transpiration in the bare soil fraction (??)
      !- First layer
      tmat(ji, 1, 1) = zero
      tmat(ji, 1, 2) = f(ji, 1)
      tmat(ji, 1, 3) = g1(ji, 1)
      rhs(ji, 1) = fp(ji, 1) * mcl(ji, 1, jst) + gp(ji, 1) * mcl(ji, 2, jst) - flux_top - (b(ji, 1) + b(ji, 2)) / deux * &
&(dt_sechiba / one_day)
      !- soil body
        DO jsl = 2, nslm - 1
        tmat(ji, jsl, 1) = e(ji, jsl)
        tmat(ji, jsl, 2) = f(ji, jsl)
        tmat(ji, jsl, 3) = g1(ji, jsl)
        rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + gp(ji, jsl) * mcl(ji, jsl + 1, jst) &
&+ (b(ji, jsl - 1) - b(ji, jsl + 1)) * (dt_sechiba / one_day) / deux
      END DO
      !- Last layer
      jsl = nslm
      tmat(ji, jsl, 1) = e(ji, jsl)
      tmat(ji, jsl, 2) = f(ji, jsl)
      tmat(ji, jsl, 3) = zero
      rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + (b(ji, jsl - 1) + b(ji, jsl) * (un - &
&deux * free_drain_coef(ji, jst))) * (dt_sechiba / one_day) / deux
      !- Store the equations for later use (9.6)
        DO jsl = 1, nslm
        srhs(ji, jsl) = rhs(ji, jsl)
        stmat(ji, jsl, 1) = tmat(ji, jsl, 1)
        stmat(ji, jsl, 2) = tmat(ji, jsl, 2)
        stmat(ji, jsl, 3) = tmat(ji, jsl, 3)
      END DO

      !! 8.6 Solve the diffusion equation, assuming that flux_top=evapot_penm (updates mcl)
      CALL hydrol_soil_tridiag_acc(ji, kjpindex, jst)

      !! 9.7 Alternative solution with mc(1)=mcr in points where the above solution leads to mcl<mcr
      ! hydrol_soil_tridiag calculates mc recursively from the top as a fonction of rhs and tmat
      ! We re-use these the above values, but for mc(1)=mcr and the related tmat

      ! by construction, mc and mcl are always on the same side of mcr, so we can use mcl here
      resolv(ji) = (mcl(ji, 1, jst) .LT. (mcr(ji)) .AND. flux_top .GT. min_sechiba)
      !! Reset the coefficient for diffusion (tridiag is only solved if resolv(ji) = .TRUE.)O
        DO jsl = 1, nslm
        !- The new condition is to put the upper layer at residual soil moisture
        rhs(ji, jsl) = srhs(ji, jsl)
        tmat(ji, jsl, 1) = stmat(ji, jsl, 1)
        tmat(ji, jsl, 2) = stmat(ji, jsl, 2)
        tmat(ji, jsl, 3) = stmat(ji, jsl, 3)
      END DO

      tmat(ji, 1, 2) = un
      tmat(ji, 1, 3) = zero
      rhs(ji, 1) = mcr(ji)

      ! Solves the diffusion equation with new surface bc where resolv=T
      CALL hydrol_soil_tridiag_acc(ji, kjpindex, jst)

      !! 8.8 In both case, we have drainage to be consistent with rhs
      flux_bottom = mask_soiltile(ji, jst) * k(ji, nslm) * free_drain_coef(ji, jst) * (dt_sechiba / one_day)

        !! 8.9 Water budget to assess the top flux = soil evaporation
        !      Where resolv=F at the 2nd step (9.6), it should simply be the potential evaporation

        ! Total soil moisture content for water budget

        DO jsl = 1, nslm
        mc(ji, jsl, jst) = MAX(mcl(ji, jsl, jst), mcl(ji, jsl, jst) + profil_froz_hydro_ns(ji, jsl, jst) * (mc(ji, jsl, jst) - &
&mcr(ji)))
        ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we get mc=mcl
      END DO

      tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * &
&mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
      END DO
      tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit

      ! Deduce upper flux from soil moisture variation and bottom flux
      ! TMCi-D-BSE=TMC (BSE=bare soil evap=TMCi-TMC-D)
      ! The numerical errors of tridiag close to saturation cannot be simply solved here,
      ! we can only hope they are not too large because we don't add water at this stage...
      evap_bare_lim_ns(ji, jst) = mask_soiltile(ji, jst) * (tmcint - tmc(ji, jst) - flux_bottom - SUM(rootsink(ji, :, jst)))

        !! 8.10 evap_bare_lim_ns is turned from an evaporation rate to a beta
        ! Here we weight evap_bare_lim_ns by the fraction of bare evaporating soil.
        ! This is given by frac_bare_ns, taking into account bare soil under vegetation
        IF (vegtot(ji) .GT. min_sechiba) THEN
        evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) * frac_bare_ns(ji, jst)
      ELSE
        evap_bare_lim_ns(ji, jst) = 0.
      END IF

        ! We divide by evapot, which is consistent with diffuco (evap_bare_lim_ns < evapot_penm/evapot)
        ! Further decrease if tmc_litter is below the wilting point

        IF (do_rsoil) THEN
        IF (evapot(ji) .GT. min_sechiba) THEN
          evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) / evapot(ji)
        ELSE
          evap_bare_lim_ns(ji, jst) = zero
          ! not redundant with the is_under_mcr case below
          ! but not necessarily useful
        END IF
        evap_bare_lim_ns(ji, jst) = MAX(MIN(evap_bare_lim_ns(ji, jst), 1.), 0.)
      ELSE
        IF ((evapot(ji) .GT. min_sechiba) .AND. (tmc_litter(ji, jst) .GT. (tmc_litter_wilt(ji, jst)))) THEN
          evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) / evapot(ji)
        ELSE IF ((evapot(ji) .GT. min_sechiba) .AND. (tmc_litter(ji, jst) .GT. (tmc_litter_res(ji, jst)))) THEN
          evap_bare_lim_ns(ji, jst) = (un / deux) * evap_bare_lim_ns(ji, jst) / evapot(ji)
          ! This is very arbitrary, with no justification from the literature
        ELSE
          evap_bare_lim_ns(ji, jst) = zero
        END IF
        evap_bare_lim_ns(ji, jst) = MAX(MIN(evap_bare_lim_ns(ji, jst), 1.), 0.)
      END IF
      IF (is_under_mcr(jst)) THEN
        evap_bare_lim_ns(ji, jst) = zero
      END IF

        !! 8.12 Restores mc, mcl, and tmc, to erase the effect of the dummy integrations
        !!      on these prognostic variables
        DO jsl = 1, nslm
        mc(ji, jsl, jst) = mask_soiltile(ji, jst) * mcint(jsl)
        mcl(ji, jsl, jst) = mask_soiltile(ji, jst) * mclint(jsl)
      END DO
      tmc(ji, jst) = temp

    END DO
    !end loop on tiles for dummy integration

    !! 9. evap_bar_lim is the grid-cell scale beta
    evap_bare_lim(ji) = SUM(evap_bare_lim_ns(ji, :) * vegtot(ji) * soiltile(ji, :))
    r_soil = SUM(r_soil_ns(:) * vegtot(ji) * soiltile(ji, :))
    ! si vegtot LE min_sechiba, evap_bare_lim_ns et evap_bare_lim valent zero


    !! 10. XIOS export of local variables, including water conservation checks
    !CALL xios_orchidee_send_field("ksat",ksat) ! mm/d (for CMIP6, once)
    !CALL xios_orchidee_send_field("psi_moy",psi_moy) ! mm (for SP-MIP)
    !CALL xios_orchidee_send_field("wtd",wtd) ! in m
    !CALL xios_orchidee_send_field("ru_corr",ru_corr/dt_sechiba)   ! adjustment flux added to surface runoff (included in runoff)
    !CALL xios_orchidee_send_field("ru_corr2",ru_corr2/dt_sechiba)
    !CALL xios_orchidee_send_field("dr_corr",dr_corr/dt_sechiba)   ! adjustment flux added to drainage (included in drainage)
    !CALL xios_orchidee_send_field("dr_corrnum",dr_corrnum/dt_sechiba)
    !CALL xios_orchidee_send_field("dr_force",dr_force/dt_sechiba) ! adjustement flux added to drainage to sustain a forced wtd
    !CALL xios_orchidee_send_field("qinfilt",qinfilt/dt_sechiba)
    !CALL xios_orchidee_send_field("ru_infilt",ru_infilt/dt_sechiba)
    !CALL xios_orchidee_send_field("r_soil",r_soil) ! s/m

    !IF (check_cwrr) THEN
    !   CALL xios_orchidee_send_field("check_infilt",check_infilt/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_tr",check_tr/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_over",check_over/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_under",check_under/dt_sechiba)
    !   ! Variables calculated in hydrol_diag_soil_flux
    !   CALL xios_orchidee_send_field("qflux",qflux/dt_sechiba) ! upward water flux at the low interface of each layer
    !   CALL xios_orchidee_send_field("check_top",check_top/dt_sechiba) !water budget residu in top layer
    !END IF


  END SUBROUTINE hydrol_soil_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil
    !!
    !>\BRIEF        This routine computes soil processes with CWRR scheme (Richards equation solved by finite differences).
    !! Note that the water fluxes are in kg/m2/dt_sechiba.
    !!
    !! DESCRIPTION  :
    !! 0. Initialisation, and split 2d variables to 3d variables, per soil tile
    !! -- START MAIN LOOP (prognostic loop to update mc and mcl) OVER SOILTILES
    !! 1. FIRSTLY, WE CHANGE MC BASED ON EXTERNAL FLUXES, ALL APPLIED AT THE SOIL SURFACE
    !! 1.1 Reduces water2infilt and water2extract to their difference
    !! 1.2 To remove water2extract (including bare soilevaporation) from top layer
    !! 1.3 Infiltration
    !! 1.4 Reinfiltration of surface runoff : compute temporary surface water and extract from runoff
    !! 2. SECONDLY, WE UPDATE MC FROM DIFFUSION, INCLUDING DRAINAGE AND ROOTSINK
    !!    This will act on mcl (liquid water content) only
    !! 2.1 K and D are recomputed after infiltration
    !! 2.2 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
    !! 2.3 We define mcl (liquid water content) based on mc and profil_froz_hydro_ns
    !! 2.4 We calculate the total SM at the beginning of the routine tridiag for water conservation check
    !! 2.5 Defining where diffusion is solved : everywhere
    !! 2.6 We define the system of linear equations for mcl redistribution
    !! 2.7 Solves diffusion equations
    !! 2.8 Computes drainage = bottom boundary condition, consistent with rhs(ji,jsl=nslm)
    !! 2.9 For water conservation check during redistribution, we calculate the total liquid SM
    !!     at the end of the routine tridiag, and we compare the difference with the flux...
    !! 3. AFTER DIFFUSION/REDISTRIBUTION
    !! 3.1 Updating mc, as all the following checks against saturation will compare mc to mcs
    !! 3.2 Correct here the possible over-saturation values (subroutine hydrol_soil_smooth_over_mcs2 acts on mc)
    !!     Here hydrol_soil_smooth_over_mcs2 discard all excess as ru_corr_ns, oriented to either ru_ns or dr_ns
    !! 3.3 Negative runoff is reported to drainage
    !! 3.4 Optional block to force saturation below zwt_force
    !! 3.5 Diagnosing the effective water table depth
    !! 3.6 Diagnose under_mcr to adapt water stress calculation below
    !! 4. At the end of the prognostic calculations, we recompute important moisture variables
    !! 4.1 Total soil moisture content (water2infilt added below)
    !! 4.2 mcl is a module variable; we update it here for calculating bare soil evaporation,
    !! 5. Optional check of the water balance of soil column (if check_cwrr)
    !! 5.1 Computation of the vertical water fluxes
    !! 6. SM DIAGNOSTICS FOR OTHER ROUTINES, MODULES, OR NEXT STEP
    !! 6.1 Total soil moisture, soil moisture at litter levels, soil wetness, us, humrelv, vesgtressv
    !! 6.2 We need to turn off evaporation when is_under_mcr
    !! 6.3 Calculate the volumetric soil moisture content (mc_layh and mcl_layh) needed in thermosoil
    !! 6.4 The hydraulic conductivities exported here are the ones used in the diffusion/redistribution
    !! -- ENDING THE MAIN LOOP ON SOILTILES
    !! 7. Summing 3d variables into 2d variables
    !! 8. XIOS export of local variables, including water conservation checks
    !! 9. COMPUTING EVAP_BARE_LIM_NS FOR NEXT TIME STEP, WITH A LOOP ON SOILTILES
    !!    The principle is to run a dummy integration of the water redistribution scheme
    !!    to check if the SM profile can sustain a potential evaporation.
    !!    If not, the dummy integration is redone from the SM profile of the end of the normal integration,
    !!    with a boundary condition leading to a very severe water limitation: mc(1)=mcr
    !! 10. evap_bar_lim is the grid-cell scale beta
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================

    SUBROUTINE hydrol_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget, veget_max, soiltile, njsc, reinf_slope_soil, &
&transpir, vevapnu, evapot, evapot_penm, runoff, drainage, returnflow, reinfiltration, irrigation, tot_melt, evap_bare_lim, &
&evap_bare_lim_ns, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, stempdiag, snow, snowdz, &
&tot_bare_soil, u, v, tq_cdrag, mc_layh, mcl_layh, mc_layh_s, mcl_layh_s, e_frac, ksoil, altmax, root_profile, root_depth, &
&root_deficit, circ_class_biomass, us, precip_rain, totfrac_nobio, frac_snow_nobio, F_absorption)

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget
    !! Fraction of vegetation type
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! Map of max vegetation types [-]
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class
    !! in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydrol_arch = .TRUE.)
    !!  @tex $(m^3 s^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: reinf_slope_soil
    !! Fraction of surface runoff that reinfiltrates per soil tile
    !!  (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: returnflow
    !! Water returning to the soil from the bottom
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: reinfiltration
    !! Water returning to the top of the soil
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: irrigation
    !! Irrigation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot
    !! Potential evaporation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evapot_penm
    !! Potential evaporation "Penman" (Milly's correction)
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_melt
    !! Total melt from snow and ice
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: snow
    !! Snow mass
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(IN) :: snowdz
    !! Snow depth (m)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    !!  (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: v
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: u
    !! Horizontal wind speed
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tq_cdrag
    !! Surface drag coefficient
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Fraction of water transpired supplied by individual layers (no units)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: altmax
    !! Maximul active layer thickness (m). Be careful, here active means non frozen.
    !! Not related with the active soil carbon pool.
    REAL(KIND = r_std), DIMENSION(:, :, :, :, :), INTENT(IN) :: circ_class_biomass
    !! Biomass components of the model tree
    !! within a circumference class
    !! class @tex $(g C ind^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rain precipitation
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: totfrac_nobio
    !! Total fraction of continental ice+lakes+...
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area


    !! 0.2 Output variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff
    !! Surface runoff
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drainage
    !! Drainage
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: evap_bare_lim
    !! Limitation factor (beta) for bare soil evaporation
    !! on each soil column (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(INOUT) :: evap_bare_lim_ns
    !! Limitation factor (beta) for bare soil evaporation
    !! on each soil column (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag
    !! Relative soil moisture in each diag soil layer
    !! with respect to (mcfc-mcw) (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: shumdiag_perma
    !! Percent of porosity filled with water (mc/mcs)
    !! in each diag soil layer (for the thermal computations)
    !! (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: k_litt
    !! Litter approximated hydraulic conductivity
    !!  @tex $(mm d^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: litterhumdiag
    !! Mean of soil_wet_litter across soil tiles
    !! (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(OUT) :: vegstress
    !! Veg. moisture stress (only for vegetation
    !! growth) (unitless, [0-1])
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drysoil_frac
    !! Function of the litter humidity
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: mc_layh
    !! Volumetric water content (liquid + ice) for each soil layer
    !! averaged over the mesh (for thermosoil)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(OUT) :: mcl_layh
    !! Volumetric liquid water content for each soil layer
    !! averaged over the mesh (for thermosoil)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: mc_layh_s
    !! Volumetric soil moisture content for each layer in hydrol(liquid + ice) [m3/m3]
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: mcl_layh_s
    !! Volumetric soil moisture content for each layer in hydrol(liquid) [m3/m3]
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof), INTENT(OUT) :: root_profile
    !! Normalized root mass/length fraction in each soil layer
    !! (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ndepths), INTENT(OUT) :: root_depth
    !! Node and interface numbers at which the deepest roots
    !! occur (1 to nslm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: root_deficit
    !! water deficit to reach SM target of soil column, for irrigation demand


    !! 0.3 Modified variables

    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: vevapnu
    !! Bare soil evaporation
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(INOUT) :: humrel
    !! Relative humidity (0-1, dimensionless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(OUT) :: ksoil
    !! Soil conductivity (a copy of k for each soil type)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)



    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: ji
    !! Indice
    INTEGER(KIND = i_std) :: jst_kfact_root
    !! Indice for kfact_root calculation
    REAL(KIND = r_std), PARAMETER :: frac_mcs = 0.66
    !! Temporary depth
    REAL(KIND = r_std), DIMENSION(kjpindex) :: temp
    !! Temporary value for fluxes
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcold
    !! Total SM at beginning of hydrol_soil (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcint
    !! Ancillary total SM (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mcint
    !! To save mc values for future use
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mclint
    !! To save mcl values for future use
    LOGICAL, DIMENSION(kjpindex, nstm) :: is_under_mcr
    !! Identifies under residual soil moisture points
    LOGICAL, DIMENSION(kjpindex) :: is_over_mcs
    !! Identifies over saturated soil moisture points
    REAL(KIND = r_std), DIMENSION(kjpindex) :: diff
    REAL(KIND = r_std), DIMENSION(kjpindex) :: deltahum
    !!
    LOGICAL(KIND = r_std), DIMENSION(kjpindex) :: test
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex) :: water2extract
    !! Water flux to be extracted at the soil surface
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: returnflow_soil
    !! Water from the routing back to the bottom of
    !! the soil @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: reinfiltration_soil
    !! Water from the routing back to the top of the
    !! soil @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: irrigation_soil
    !! Water from irrigation returning to soil moisture per soil tile
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_bottom
    !! Flux at bottom of the soil column
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_top
    !! Flux at top of the soil column (for bare soil evap)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: qinfilt_ns
    !! Effective infiltration flux per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: qinfilt
    !! Effective infiltration flux
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: ru_infilt_ns
    !! Surface runoff from hydrol_soil_infilt per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ru_infilt
    !! Surface runoff from hydrol_soil_infilt
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: ru_corr_ns
    !! Surface runoff produced to correct excess per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ru_corr
    !! Surface runoff produced to correct excess
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: ru_corr2_ns
    !! Correction of negative surface runoff per soil tile
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ru_corr2
    !! Correction of negative surface runoff
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: dr_corr_ns
    !! Drainage produced to correct excess
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: dr_corrnum_ns
    !! Drainage produced to correct numerical errors in tridiag
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dr_corr
    !! Drainage produced to correct excess
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dr_corrnum
    !! Drainage produced to correct numerical errors in tridiag
    !!  @tex $(kg m^{-2} dt\_sechiba^{-1})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: dmc
    !! Delta mc when forcing saturation (zwt_force)
    !!  @tex $(m^{3} m^{-3})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: dr_force_ns
    !! Delta drainage when forcing saturation (zwt_force)
    !!  per soil tile  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dr_force
    !! Delta drainage when forcing saturation (zwt_force)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: wtd_ns
    !! Effective water table depth (m)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: wtd
    !! Mean water table depth in the grid-cell (m)

    ! For the calculation of soil_wet_ns and us/humrel/vegstress
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: sm
    !! Soil moisture of each layer (liquid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: smt
    !! Soil moisture of each layer (liquid+solid phase)
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: smw
    !! Soil moisture of each layer at wilting point
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: smf
    !! Soil moisture of each layer at field capacity
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: sms
    !! Soil moisture of each layer at saturation
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: sm_nostress
    !! Soil moisture of each layer at which us reaches 1
    !!  @tex $(kg m^{-2})$ @endtex
    ! For water conservation checks (in mm/dtstep unless otherwise mentioned)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check_infilt_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check1_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check_tr_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check_over_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check_under_ns
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! Total soil moisture at beginning of routine (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! Total soil moisture at end of routine (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: diag_tr
    !! Transpiration flux
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_infilt
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check1
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_tr
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_over
    !! Water conservation diagnostic at routine scale
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_under
    !! Water conservation diagnostic at routine scale
    ! For irrigation triggering
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: lai_irrig_trig
    !! Number of PFT per cell with LAI> LAI_IRRIG_MIN -
    ! Diagnostic of the vertical soil water fluxes
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: qflux
    !! Local upward flux into soil layer
    !! from lower interface
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_top
    !! Water budget residu in top soil layer
    !!  @tex $(kg m^{-2})$ @endtex

    ! Variables for calculation of a soil resistance, option do_rsoil (following the formulation of Sellers et al 1992, implemented
    !& in Oleson et al. 2008)
    REAL(KIND = r_std) :: speed
    !! magnitude of wind speed required for Aerodynamic resistance
    REAL(KIND = r_std) :: ra
    !! diagnosed aerodynamic resistance
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mc_rel
    !! first layer relative soil moisture, required for rsoil
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evap_soil
    !! soil evaporation from Oleson et al 2008
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: r_soil_ns
    !! soil resistance from Oleson et al 2008
    REAL(KIND = r_std), DIMENSION(kjpindex) :: r_soil
    !! soil resistance from Oleson et al 2008
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcs_litter
    !! Saturated soil moisture in the 4 "litter" soil layers
    REAL(KIND = r_std), DIMENSION(nslm) :: root_profile_tmp
    !! Temporary variable to calculate the root_profile

    ! For CMIP6 and SP-MIP : ksat and matric pressure head psi(theta)
    REAL(KIND = r_std) :: avg
    REAL(KIND = r_std) :: mvg
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: psi
    !! Matric head (per soil layer and soil tile) [mm=kg/m2]
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: psi_moy
    !! Mean matric head per soil layer [mm=kg/m2]
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: ksat
    !! Saturated hydraulic conductivity at each node (mm/d)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof) :: tmp
    !! temporary variable for writing the root profiles to XIOS

    !_
    !& 
!& ================================================================================================================================

    !! 0.1 Arrays with DIMENSION(kjpindex)

    returnflow_soil(:) = zero
    reinfiltration_soil(:) = zero
    irrigation_soil(:, :) = zero
    qflux_ns(:, :, :) = zero
    mc_layh(:, :) = zero
    ! for thermosoil
    mcl_layh(:, :) = zero
    ! for thermosoil
    kk(:, :, :) = zero
    kk_moy(:, :) = zero
    undermcr(:) = zero
    ! needs to be initialized outside from jst loop
    ksat(:, :) = zero
    psi_moy(:, :) = zero

      !! Calculate kfact_root
      IF (kfact_root_const) THEN
      kfact_root(:, :, :) = un
    ELSE
      !! An exponential factor is used to increase ks near the surface depending on the amount of roots in the soil
      !! through a geometric average over the vegets
      !! This comes from the PhD thesis of d'Orgeval, 2006, p82; d'Orgeval et al. 2008, Eqs. 3-4
      !! (Calibrated against Hapex-Sahel measurements)
      !! Since rev 2916: veget_max/2 is used instead of veget
      kfact_root(:, :, :) = un
      DO jsl = 1, nslm
        DO jv = 2, nvm
          jst_kfact_root = pref_soil_veg(jv)
          DO ji = 1, kjpindex
            IF (soiltile(ji, jst_kfact_root) .GT. min_sechiba) THEN
              kfact_root(ji, jsl, jst_kfact_root) = kfact_root(ji, jsl, jst_kfact_root) * MAX((MAXVAL(ks_usda) / ks(ji)) ** (- &
&vegetmax_soil(ji, jv, jst_kfact_root) / 2 * (humcste(jv) * zz(jsl) / mille - un) / deux), un)
            END IF
          END DO
        END DO
      END DO
    END IF



      IF (ok_freeze_cwrr) THEN

        ! 0.1 Calculate the temperature and fozen fraction at the hydrological levels
        ! Calculates profil_froz_hydro_ns as a function of stempdiag and mc if ok_thermodynamical_freezing
        ! These values will be kept till the end of the prognostic loop
        DO jst = 1, nstm
        CALL hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, jst, njsc, stempdiag)
      END DO

    ELSE

      profil_froz_hydro_ns(:, :, :) = zero

    END IF

    !! 0.2 Split 2d variables to 3d variables, per soil tile
    !  Here, the evaporative fluxes are distributed over the soiltiles as a function of the
    !    corresponding control factors; they are normalized to vegtot
    !  At step 7, the reverse transformation is used for the fluxes produced in hydrol_soil
    !    flux_cell(ji)=sum(flux_ns(ji,jst)*soiltile(ji,jst)*vegtot(ji))


    CALL hydrol_split_soil(kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, &
&tot_bare_soil, us, e_frac, F_absorption)


      !! 0.3 Common variables related to routing, with all return flow applied to the soil surface
      ! The fluxes coming from the routing are uniformly splitted into the soiltiles,
      !    but are normalized to vegtot like the above fluxes:
      !            flux_ns(ji,jst)=flux_cell(ji)/vegtot(ji)
      ! It is the case for : irrigation_soil(ji) and reinfiltration_soil(ji) cf below
      ! It is also the case for subsinksoil(ji), which is divided by (1-tot_frac_nobio) at creation in hydrol_snow
      ! AD16*** The transformation in 0.2 and 0.3 is likely to induce conservation problems
      !         when tot_frac_nobio NE 0, since sum(soiltile) NE vegtot in this case
      IF (.NOT. old_irrig_scheme) THEN
      !
        IF (.NOT. irrigated_soiltile) THEN
        DO ji = 1, kjpindex
          IF (vegtot(ji) .GT. min_sechiba) THEN
            returnflow_soil(ji) = zero
            reinfiltration_soil(ji) = (returnflow(ji) + reinfiltration(ji)) / vegtot(ji)
            IF (soiltile(ji, irrig_st) .GT. min_sechiba) THEN
              !irrigation_soil(ji, 1:2) = 0, if irrig_st = 3. Not put because Values
              !are already zero, due to initialization
              irrigation_soil(ji, irrig_st) = irrigation(ji) / (soiltile(ji, irrig_st) * vegtot(ji))
              !Irrigation is kg/m2 of grid cell. Here, all that water is put on
              !irrig_st (irrigated soil tile), by default = 3, for the others
              !soil tiles irrigation = zero
            END IF
          END IF
        END DO
      END IF
    ELSE
      !
        DO ji = 1, kjpindex
        IF (vegtot(ji) .GT. min_sechiba) THEN
          ! returnflow_soil is assumed to enter from the bottom, but it is not possible with CWRR
          returnflow_soil(ji) = zero
          reinfiltration_soil(ji) = (returnflow(ji) + reinfiltration(ji)) / vegtot(ji)
          irrigation_soil(ji, :) = irrigation(ji) / vegtot(ji)
          ! irrigation_soil(ji) = irrigation(ji)/vegtot(ji)
          ! Computed for all the grid cell. New way is equivalent, and coherent
          ! with irrigation_soil new dimensions (cells, soil tiles)
          ! Irrigation is kg/m2 of grid cell. For the old irrig. scheme,
          ! irrigation soil is the same for every soil tile
          ! Next lines are in tag 2.0, deleted because values are already init to zero
          ! ELSE
          ! returnflow_soil(ji) = zero
          ! reinfiltration_soil(ji) = zero
          ! irrigation_soil(ji) = zero
          ! ENDIF
        END IF
      END DO
    END IF

      !! -- START MAIN LOOP (prognostic loop to update mc and mcl) OVER SOILTILES
      !!    The called subroutines work on arrays with DIMENSION(kjpindex),
      !!    recursively used for each soiltile jst

      DO jst = 1, nstm

      is_under_mcr(:, jst) = .FALSE.
      is_over_mcs(:) = .FALSE.

      !! 0.4. Keep initial values for future check-up

      ! Total moisture content (including water2infilt) is saved for balance checks at the end
      ! In hydrol_tmc_update, tmc is increased by water2infilt(ji,jst), but mc is not modified !
      tmcold(:) = tmc(:, jst)

        ! The value of mc is kept in mcint (nstm dimension removed), in case needed for water balance checks
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mcint(ji, jsl) = mask_soiltile(ji, jst) * mc(ji, jsl, jst)
        END DO
      END DO
      !
        ! Initial total moisture content : tmcint does not include water2infilt, contrarily to tmcold
        DO ji = 1, kjpindex
        tmcint(ji) = dz(2) * (trois * mcint(ji, 1) + mcint(ji, 2)) / huit
      END DO
      DO jsl = 2, nslm - 1
        DO ji = 1, kjpindex
          tmcint(ji) = tmcint(ji) + dz(jsl) * (trois * mcint(ji, jsl) + mcint(ji, jsl - 1)) / huit + dz(jsl + 1) * (trois * &
&mcint(ji, jsl) + mcint(ji, jsl + 1)) / huit
        END DO
      END DO
      DO ji = 1, kjpindex
        tmcint(ji) = tmcint(ji) + dz(nslm) * (trois * mcint(ji, nslm) + mcint(ji, nslm - 1)) / huit
      END DO

        !! 1. FIRSTLY, WE CHANGE MC BASED ON EXTERNAL FLUXES, ALL APPLIED AT THE SOIL SURFACE
        !!   Input = water2infilt(ji,jst) + irrigation_soil(ji) + reinfiltration_soil(ji) + precisol_ns(ji,jst)
        !!      - negative evaporation fluxes (MIN(ae_ns(ji,jst),zero)+ MIN(subsinksoil(ji),zero))
        !!   Output = MAX(ae_ns(ji,jst),zero) + subsinksoil(ji) = positive evaporation flux = water2extract
        ! In practice, negative subsinksoil(ji) is not possible

        !! 1.1 Reduces water2infilt and water2extract to their difference

        ! Compares water2infilt and water2extract to keep only difference
        ! Here, temp is used as a temporary variable to store the min of water to infiltrate vs evaporate
        DO ji = 1, kjpindex
        temp(ji) = MIN(water2infilt(ji, jst) + irrigation_soil(ji, jst) + reinfiltration_soil(ji) - MIN(ae_ns(ji, jst), zero) - &
&MIN(subsinksoil(ji), zero) + precisol_ns(ji, jst), MAX(ae_ns(ji, jst), zero) + MAX(subsinksoil(ji), zero))
      END DO

        ! The water to infiltrate at the soil surface is either 0, or the difference to what has to be evaporated
        !   - the initial water2infilt (right hand side) results from qsintveg changes with vegetation updates
        !   - irrigation_soil is the input flux to the soil surface from irrigation
        !   - reinfiltration_soil is the input flux to the soil surface from routing 'including returnflow)
        !   - eventually, water2infilt holds all fluxes to the soil surface except precisol (reduced by water2extract)
        DO ji = 1, kjpindex
        !Note that in tag 2.0, irrigation_soil(ji), changed to be coherent with new variable dimension
        water2infilt(ji, jst) = water2infilt(ji, jst) + irrigation_soil(ji, jst) + reinfiltration_soil(ji) - MIN(ae_ns(ji, jst), &
&zero) - MIN(subsinksoil(ji), zero) + precisol_ns(ji, jst) - temp(ji)
      END DO

        ! The water to evaporate from the sol surface is either the difference to what has to be infiltrated, or 0
        !   - subsinksoil is the residual from sublimation is the snowpack is not sufficient
        !   - how are the negative values of ae_ns taken into account ???
        DO ji = 1, kjpindex
        water2extract(ji) = MAX(ae_ns(ji, jst), zero) + MAX(subsinksoil(ji), zero) - temp(ji)
      END DO

      ! Here we acknowledge that subsinksoil is part of ae_ns, but ae_ns is not used further
      ae_ns(:, jst) = ae_ns(:, jst) + subsinksoil(:)

      !! 1.2 To remove water2extract (including bare soil) from top layer
      flux_top(:) = water2extract(:)

        !! 1.3 Infiltration

        !! Definition of flux_infilt
        DO ji = 1, kjpindex
        ! Initialise the flux to be infiltrated
        flux_infilt(ji) = water2infilt(ji, jst)
      END DO

      !! K and D are computed for the profile of mc before infiltration
      !! They depend on the fraction of soil ice, given by profil_froz_hydro_ns
      CALL hydrol_soil_coef(mcr, mcs, kjpindex, jst, njsc)

      !! Infiltration and surface runoff are computed
      !! Infiltration stems from comparing liquid water2infilt to initial total mc (liquid+ice)
      !! The conductivity comes from hydrol_soil_coef and relates to the liquid phase only
      !  This seems consistent with ok_freeze
      CALL hydrol_soil_infilt(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, jst, njsc, flux_infilt, stempdiag, qinfilt_ns, &
&ru_infilt_ns, check_infilt_ns)
      ru_ns(:, jst) = ru_infilt_ns(:, jst)

        !! 1.4 Reinfiltration of surface runoff : compute temporary surface water and extract from runoff
        ! Evrything here is liquid
        ! RK: water2infilt is both a volume for future reinfiltration (in mm) and a correction term for surface runoff (in
        !& mm/dt_sechiba)
        IF (.NOT. doponds) THEN
        ! this is the general case...
          DO ji = 1, kjpindex
          water2infilt(ji, jst) = reinf_slope_soil(ji, jst) * ru_ns(ji, jst)
        END DO
      ELSE
        DO ji = 1, kjpindex
          water2infilt(ji, jst) = zero
        END DO
      END IF
      !
        DO ji = 1, kjpindex
        ru_ns(ji, jst) = ru_ns(ji, jst) - water2infilt(ji, jst)
      END DO

      !! 2. SECONDLY, WE UPDATE MC FROM DIFFUSION, INCLUDING DRAINAGE AND ROOTSINK
      !!    This will act on mcl only

      !! 2.1 K and D are recomputed after infiltration
      !! They depend on the fraction of soil ice, still given by profil_froz_hydro_ns
      CALL hydrol_soil_coef(mcr, mcs, kjpindex, jst, njsc)

      !! 2.2 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
      !! This process will further act on mcl only, based on a, b, d from hydrol_soil_coef
      CALL hydrol_soil_setup(kjpindex, jst)

        !! 2.3 We define mcl (liquid water content) based on mc and profil_froz_hydro_ns
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
          ! we always have mcl<=mc
          ! if mc>mcr, then mcl>mcr; if mc=mcr,mcl=mcr; if mc<mcr, then mcl<mcr
          ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
        END DO
      END DO

        ! The value of mcl is kept in mclint (nstm dimension removed), used in the flux computation after diffusion
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mclint(ji, jsl) = mask_soiltile(ji, jst) * mcl(ji, jsl, jst)
        END DO
      END DO

        !! 2.3bis Diagnostic of the matric potential used for redistribution by Richards/tridiag (in m)
        !  We use VG relationship giving psi as a function of mc (mcl in our case)
        !  With patches against numerical pbs when (mc_ratio - un) becomes very slightly negative (gives NaN)
        !  or if psi become too strongly negative (pbs with xios output)
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          IF (soiltile(ji, jst) .GT. zero) THEN
            mvg = un - un / nvan_mod_tab(jsl, ji)
            avg = avan_mod_tab(jsl, ji) * 1000.
            ! to convert in m-1
            mc_ratio = MAX(10. ** (- 14 * mvg), (mcl(ji, jsl, jst) - mcr(ji)) / (mcs(ji) - mcr(ji))) ** (- un / mvg)
            psi = - MAX(zero, (mc_ratio - un)) ** (un / nvan_mod_tab(jsl, ji)) / avg
            ! in m
            psi_moy(ji, jsl) = psi_moy(ji, jsl) + soiltile(ji, jst) * psi
            ! average across soil tiles
          END IF
        END DO
      END DO

      !! 2.4 We calculate the total SM at the beginning of the routine tridiag for water conservation check
      !  (on mcl only, since the diffusion only modifies mcl)
      tmci(:) = dz(2) * (trois * mcl(:, 1, jst) + mcl(:, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mcl(:, jsl, jst) + mcl(:, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(:, &
&jsl, jst) + mcl(:, jsl + 1, jst)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mcl(:, nslm, jst) + mcl(:, nslm - 1, jst)) / huit

      !! 2.5 Defining where diffusion is solved : everywhere
      !! Since mc>mcs is not possible after infiltration, and we accept that mc<mcr
      !! (corrected later by shutting off all evaporative fluxes in this case)
      !  Nothing is done if resolv=F
      resolv(:) = (mask_soiltile(:, jst) .GT. 0)

        !! 2.6 We define the system of linear equations for mcl redistribution,
        !! based on the matrix coefficients from hydrol_soil_setup
        !! following the PhD thesis of de Rosnay (1999), p155-157
        !! The bare soil evaporation (subtracted from infiltration) is used directly as flux_top
        ! rhs for right-hand side term; fp for f'; gp for g'; ep for e'; with flux=0 !

        !- First layer
        DO ji = 1, kjpindex
        tmat(ji, 1, 1) = zero
        tmat(ji, 1, 2) = f(ji, 1)
        tmat(ji, 1, 3) = g1(ji, 1)
        rhs(ji, 1) = fp(ji, 1) * mcl(ji, 1, jst) + gp(ji, 1) * mcl(ji, 2, jst) - flux_top(ji) - (b(ji, 1) + b(ji, 2)) / deux * &
&(dt_sechiba / one_day) - rootsink(ji, 1, jst)
      END DO
      !- soil body
        DO jsl = 2, nslm - 1
        DO ji = 1, kjpindex
          tmat(ji, jsl, 1) = e(ji, jsl)
          tmat(ji, jsl, 2) = f(ji, jsl)
          tmat(ji, jsl, 3) = g1(ji, jsl)
          rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + gp(ji, jsl) * mcl(ji, jsl + 1, &
&jst) + (b(ji, jsl - 1) - b(ji, jsl + 1)) * (dt_sechiba / one_day) / deux - rootsink(ji, jsl, jst)
        END DO
      END DO
      !- Last layer, including drainage
        DO ji = 1, kjpindex
        jsl = nslm
        tmat(ji, jsl, 1) = e(ji, jsl)
        tmat(ji, jsl, 2) = f(ji, jsl)
        tmat(ji, jsl, 3) = zero
        rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + (b(ji, jsl - 1) + b(ji, jsl) * (un &
&- deux * free_drain_coef(ji, jst))) * (dt_sechiba / one_day) / deux - rootsink(ji, jsl, jst)
      END DO
      !- Store the equations in case needed again
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          srhs(ji, jsl) = rhs(ji, jsl)
          stmat(ji, jsl, 1) = tmat(ji, jsl, 1)
          stmat(ji, jsl, 2) = tmat(ji, jsl, 2)
          stmat(ji, jsl, 3) = tmat(ji, jsl, 3)
        END DO
      END DO

      !! 2.7 Solves diffusion equations, but only in grid-cells where resolv is true, i.e. everywhere (cf 2.2)
      !!     The result is an updated mcl profile

      CALL hydrol_soil_tridiag(kjpindex, jst)

        !! 2.8 Computes drainage = bottom boundary condition, consistent with rhs(ji,jsl=nslm)
        ! dr_ns in mm/dt_sechiba, from k in mm/d
        ! This should be done where resolv=T, like tridiag (drainage is part of the linear system !)
        DO ji = 1, kjpindex
        IF (resolv(ji)) THEN
          dr_ns(ji, jst) = mask_soiltile(ji, jst) * k(ji, nslm) * free_drain_coef(ji, jst) * (dt_sechiba / one_day)
        ELSE
          dr_ns(ji, jst) = zero
        END IF
      END DO

      !! 2.9 For water conservation check during redistribution AND CORRECTION,
      !!     we calculate the total liquid SM at the end of the routine tridiag
      tmcf(:) = dz(2) * (trois * mcl(:, 1, jst) + mcl(:, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mcl(:, jsl, jst) + mcl(:, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(:, &
&jsl, jst) + mcl(:, jsl + 1, jst)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mcl(:, nslm, jst) + mcl(:, nslm - 1, jst)) / huit

        !! And we compare the difference with the flux...
        ! Normally, tcmf=tmci-flux_top(ji)-transpir-dr_ns
        DO ji = 1, kjpindex
        diag_tr(ji) = SUM(rootsink(ji, :, jst))
      END DO
      ! Here, check_tr_ns holds the inaccuracy during the redistribution phase
      check_tr_ns(:, jst) = tmcf(:) - (tmci(:) - flux_top(:) - dr_ns(:, jst) - diag_tr(:))

        !! We solve here the numerical errors that happen when the soil is close to saturation
        !! and drainage very high, and which lead to negative check_tr_ns: the soil dries more
        !! than what is demanded by the fluxes, so we need to increase the fluxes.
        !! This is done by increasing the drainage.
        !! There are also instances of positive check_tr_ns, larger when the drainage is high
        !! They are similarly corrected by a decrease of dr_ns, in the limit of keeping a positive drainage.
        DO ji = 1, kjpindex
        IF (check_tr_ns(ji, jst) .LT. zero) THEN
          dr_corrnum_ns(ji, jst) = - check_tr_ns(ji, jst)
        ELSE
          dr_corrnum_ns(ji, jst) = - MIN(dr_ns(ji, jst), check_tr_ns(ji, jst))
        END IF
        dr_ns(ji, jst) = dr_ns(ji, jst) + dr_corrnum_ns(ji, jst)
        ! dr_ns increases/decrease if check_tr negative/positive
      END DO
      !! For water conservation check during redistribution
        IF (check_cwrr) THEN
        check_tr_ns(:, jst) = tmcf(:) - (tmci(:) - flux_top(:) - dr_ns(:, jst) - diag_tr(:))
      END IF

        !! 3. AFTER DIFFUSION/REDISTRIBUTION

        !! 3.1 Updating mc, as all the following checks against saturation will compare mc to mcs
        !      The frozen fraction is constant, so that any water flux to/from a layer changes
        !      both mcl and the ice amount. The assumption behind this is that water entering/leaving
        !      a soil layer immediately freezes/melts with the proportion profil_froz_hydro_ns/(1-profil_...)
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mc(ji, jsl, jst) = MAX(mcl(ji, jsl, jst), mcl(ji, jsl, jst) + profil_froz_hydro_ns(ji, jsl, jst) * (mc(ji, jsl, jst) - &
&mcr(ji)))
          ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we get mc=mcl
        END DO
      END DO

      !! 3.2 Correct here the possible over-saturation values (subroutine hydrol_soil_smooth_over_mcs2 acts on mc)
      !    Oversaturation results from numerical inaccuracies and can be frequent if free_drain_coef=0
      !    Here hydrol_soil_smooth_over_mcs2 discard all excess as ru_corr_ns, oriented to either ru_ns or dr_ns
      !    The former routine hydrol_soil_smooth_over_mcs, which keeps most of the excess in the soiltile
      !    after smoothing, first downward then upward, is kept in the module but not used here
      dr_corr_ns(:, jst) = zero
      ru_corr_ns(:, jst) = zero
      CALL hydrol_soil_smooth_over_mcs2(mcs, kjpindex, jst, njsc, is_over_mcs, ru_corr_ns, check_over_ns)

        ! In absence of freezing, if F is large enough, the correction of oversaturation is sent to drainage
        DO ji = 1, kjpindex
        IF ((free_drain_coef(ji, jst) .GE. 0.5) .AND. (.NOT. ok_freeze_cwrr)) THEN
          dr_corr_ns(ji, jst) = ru_corr_ns(ji, jst)
          ru_corr_ns(ji, jst) = zero
        END IF
      END DO
      dr_ns(:, jst) = dr_ns(:, jst) + dr_corr_ns(:, jst)
      ru_ns(:, jst) = ru_ns(:, jst) + ru_corr_ns(:, jst)

      !! 3.3 Negative runoff is reported to drainage
      !  Since we computed ru_ns directly from hydrol_soil_infilt, ru_ns should not be negative

      ru_corr2_ns(:, jst) = zero
      DO ji = 1, kjpindex
        IF (ru_ns(ji, jst) .LT. zero) THEN
          IF (printlev >= 3) WRITE(numout, *) 'NEGATIVE RU_NS: runoff and drainage before correction', ru_ns(ji, jst), dr_ns(ji, &
&jst)
          dr_ns(ji, jst) = dr_ns(ji, jst) + ru_ns(ji, jst)
          ru_corr2_ns(ji, jst) = - ru_ns(ji, jst)
          ru_ns(ji, jst) = 0.
        END IF
      END DO

        !! 3.4.1 Optional nudging for soil moisture
        IF (ok_nudge_mc) THEN
        CALL hydrol_nudge_mc(kjpindex, jst, mc)
      END IF


        !! 3.4.2 Optional block to force saturation below zwt_force
        ! This block is not compatible with freezing; in this case, mcl must be corrected too
        ! We test if zwt_force(1,jst) <= zmaxh, to avoid steps 1 and 2 if unnecessary

        IF (zwt_force(1, jst) <= zmaxh) THEN

          !! We force the nodes below zwt_force to be saturated
          !  As above, we compare mc to mcs
          DO jsl = 1, nslm
          DO ji = 1, kjpindex
            dmc(ji, jsl) = zero
            IF ((zz(jsl) >= zwt_force(ji, jst) * mille)) THEN
              dmc(ji, jsl) = mcs(ji) - mc(ji, jsl, jst)
              ! addition to reach mcs (m3/m3) = positive value
              mc(ji, jsl, jst) = mcs(ji)
            END IF
          END DO
        END DO

          !! To ensure conservation, this needs to be balanced by a negative change in drainage (in kg/m2/dt)
          DO ji = 1, kjpindex
          dr_force_ns(ji, jst) = dz(2) * (trois * dmc(ji, 1) + dmc(ji, 2)) / huit
          ! top layer = initialization
        END DO
        DO jsl = 2, nslm - 1
          ! intermediate layers
            DO ji = 1, kjpindex
            dr_force_ns(ji, jst) = dr_force_ns(ji, jst) + dz(jsl) * (trois * dmc(ji, jsl) + dmc(ji, jsl - 1)) / huit + dz(jsl + 1) &
&* (trois * dmc(ji, jsl) + dmc(ji, jsl + 1)) / huit
          END DO
        END DO
        DO ji = 1, kjpindex
          dr_force_ns(ji, jst) = dr_force_ns(ji, jst) + dz(nslm) * (trois * dmc(ji, nslm) + dmc(ji, nslm - 1)) / huit
          ! bottom layer
          dr_ns(ji, jst) = dr_ns(ji, jst) - dr_force_ns(ji, jst)
          ! dr_force_ns is positive and dr_ns must be reduced
        END DO

      ELSE

        dr_force_ns(:, jst) = zero

      END IF

        !! 3.5 Diagnosing the effective water table depth:
        !!     Defined as as the smallest jsl value when mc(jsl) is no more at saturation (mcs), starting from the bottom
        !      If there is a part of the soil which is saturated but underlain with unsaturated nodes,
        !      this is not considered as a water table
        DO ji = 1, kjpindex
        wtd_ns(ji, jst) = undef_sechiba
        ! in meters
        jsl = nslm
        DO WHILE ((mc(ji, jsl, jst) .EQ. mcs(ji)) .AND. (jsl > 1))
          wtd_ns(ji, jst) = zz(jsl) / mille
          ! in meters
          jsl = jsl - 1
        END DO
      END DO

      !! 3.6 Diagnose under_mcr to adapt water stress calculation below
      !      This routine does not change tmc but decides where we should turn off ET to prevent further mc decrease
      !      Like above, the tests are made on total mc, compared to mcr
      CALL hydrol_soil_smooth_under_mcr(mcr, kjpindex, jst, njsc, is_under_mcr, check_under_ns)

        !! 4. At the end of the prognostic calculations, we recompute important moisture variables

        !! 4.1 Total soil moisture content (water2infilt added below)
        DO ji = 1, kjpindex
        tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      END DO
      DO jsl = 2, nslm - 1
        DO ji = 1, kjpindex
          tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois &
&* mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
        END DO
      END DO
      DO ji = 1, kjpindex
        tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      END DO

        !! 4.2 mcl is a module variable; we update it here for calculating bare soil evaporation,
        !!     and in case we would like to export it (xios)
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
          ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
        END DO
      END DO

        !! 5. Optional check of the water balance of soil column (if check_cwrr)

        IF (check_cwrr) THEN

        !! 5.1 Computation of the vertical water fluxes and water balance of the top layer
        CALL hydrol_diag_soil_flux(kjpindex, jst, mclint, flux_top)

      END IF

        !! 6. SM DIAGNOSTICS FOR OTHER ROUTINES, MODULES, OR NEXT STEP
        !    Starting here, mc and mcl should not change anymore

        !! 6.1 Total soil moisture, soil moisture at litter levels, soil wetness, us, humrelv, vesgtressv
        !!     (based on mc)

        !! In output, tmc includes water2infilt(ji,jst)
        DO ji = 1, kjpindex
        tmc(ji, jst) = tmc(ji, jst) + water2infilt(ji, jst)
      END DO

        ! The litter is the 4 top levels of the soil
        ! Compute various field of soil moisture for the litter (used for stomate and for albedo)
        ! We exclude the frozen water from the calculation
        DO ji = 1, kjpindex
        tmc_litter(ji, jst) = dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      END DO
      ! sum from level 1 to 4
        DO jsl = 2, 4
        DO ji = 1, kjpindex
          tmc_litter(ji, jst) = tmc_litter(ji, jst) + dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + &
&dz(jsl + 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit
        END DO
      END DO

        ! Subsequent calculation of soil_wet_litter (tmc-tmcw)/(tmcfc-tmcw)
        ! Based on liquid water content
        DO ji = 1, kjpindex
        soil_wet_litter(ji, jst) = MIN(un, MAX(zero, (tmc_litter(ji, jst) - tmc_litter_wilt(ji, jst)) / (tmc_litter_field(ji, jst) &
&- tmc_litter_wilt(ji, jst))))
      END DO

      ! Preliminary calculation of various soil moistures (for each layer, in kg/m2)
      sm(:, 1) = dz(2) * (trois * mcl(:, 1, jst) + mcl(:, 2, jst)) / huit
      smt(:, 1) = dz(2) * (trois * mc(:, 1, jst) + mc(:, 2, jst)) / huit
      smw(:, 1) = dz(2) * (quatre * mcw(:)) / huit
      smf(:, 1) = dz(2) * (quatre * mcfc(:)) / huit
      sms(:, 1) = dz(2) * (quatre * mcs(:)) / huit
      DO jsl = 2, nslm - 1
        sm(:, jsl) = dz(jsl) * (trois * mcl(:, jsl, jst) + mcl(:, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(:, jsl, jst) &
&+ mcl(:, jsl + 1, jst)) / huit
        smt(:, jsl) = dz(jsl) * (trois * mc(:, jsl, jst) + mc(:, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, jst) + &
&mc(:, jsl + 1, jst)) / huit
        smw(:, jsl) = dz(jsl) * (quatre * mcw(:)) / huit + dz(jsl + 1) * (quatre * mcw(:)) / huit
        smf(:, jsl) = dz(jsl) * (quatre * mcfc(:)) / huit + dz(jsl + 1) * (quatre * mcfc(:)) / huit
        sms(:, jsl) = dz(jsl) * (quatre * mcs(:)) / huit + dz(jsl + 1) * (quatre * mcs(:)) / huit
      END DO
      sm(:, nslm) = dz(nslm) * (trois * mcl(:, nslm, jst) + mcl(:, nslm - 1, jst)) / huit
      smt(:, nslm) = dz(nslm) * (trois * mc(:, nslm, jst) + mc(:, nslm - 1, jst)) / huit
      smw(:, nslm) = dz(nslm) * (quatre * mcw(:)) / huit
      smf(:, nslm) = dz(nslm) * (quatre * mcfc(:)) / huit
      sms(:, nslm) = dz(nslm) * (quatre * mcs(:)) / huit
      ! sm_nostress = soil moisture of each layer at which us reaches 1, here at the middle of [smw,smf]
        DO jsl = 1, nslm
        sm_nostress(:, jsl) = smw(:, jsl) + pcent(njsc(:)) * (smf(:, jsl) - smw(:, jsl))
      END DO

      ! Saturated litter soil moisture for rsoil
      tmcs_litter(:) = zero
      DO jsl = 1, 4
        tmcs_litter(:) = tmcs_litter(:) + sms(:, jsl)
      END DO

        ! Here we compute root zone deficit, to have an estimate of water demand in irrigated soil column (i.e. crop and grass)
        IF (jst .EQ. irrig_st) THEN
        !It computes water deficit only on the root zone, and only on the layers where
          !there is actually a deficit. If there is not deficit, it does not take into account that layer
          DO ji = 1, kjpindex

          root_deficit(ji) = SUM(MAX(zero, beta_irrig * smf(ji, 1 : nslm_root(ji)) - sm(ji, 1 : nslm_root(ji)))) - &
&water2infilt(ji, jst)

          root_deficit(ji) = MAX(root_deficit(ji), zero)

        END DO
        !It COUNTS the number of pft with LAI > lai_irrig_min, inside the soiltile
        !It compares veget, but it is the same as they are related by a function
        lai_irrig_trig(:) = 0

          DO jv = 1, nvm
          IF (.NOT. natural(jv)) THEN
            DO ji = 1, kjpindex

                IF (veget(ji, jv) > veget_max(ji, jv) * (un - EXP(- lai_irrig_min * ext_coeff_vegetfrac(jv)))) THEN

                lai_irrig_trig(ji) = lai_irrig_trig(ji) + 1

              END IF

            END DO
          END IF

        END DO
        !If any of the PFT inside the soil tile have LAI >  lai_irrig_min (I.E. lai_irrig_trig(ji) = 0 )
          !The root deficit is set to zero, and irrigation is not triggered
          DO ji = 1, kjpindex

            IF (lai_irrig_trig(ji) < 1) THEN
            root_deficit(ji) = zero
          END IF

        END DO
      END IF

        ! Soil wetness profiles (W-Ww)/(Ws-Ww)
        ! soil_wet_ns is the ratio of available soil moisture to max available soil moisture
        ! (ie soil moisture at saturation minus soil moisture at wilting point).
        ! soil wet is a water stress for stomate, to control C decomposition
        ! Based on liquid water content
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          soil_wet_ns(ji, jsl, jst) = MIN(un, MAX(zero, (sm(ji, jsl) - smw(ji, jsl)) / (sms(ji, jsl) - smw(ji, jsl))))
        END DO
      END DO

      ! Compute us and the new humrelv to use in sechiba (with loops on the vegetation types)
      ! This is the water stress for transpiration (diffuco) and photosynthesis (diffuco)
      ! humrel is never used in stomate
      ! Based on liquid water content

      ! -- PFT1
      humrelv(:, 1, jst) = zero
      ! -- Top layer
        DO jv = 2, nvm
        DO ji = 1, kjpindex
          !- Here we make the assumption that roots do not take water from the 1st layer.
          us(ji, jv, jst, 1) = zero
          humrelv(ji, jv, jst) = zero
          ! initialisation of the sum
        END DO
      END DO

      ! There are two different ways of looking at a root profile in the code. It
      ! could reflect "structure" or "function". The code uses a different
      ! root profile depending on what it is used for. A structural and functional
      ! root profile are calculated below.
      CALL hydrol_root_profile(kjpindex, altmax, sm, smw, root_profile, root_depth)

      ! Make root_dens XIOS proof. Use NAN instead of zero to obtain the correct mean
      ! value for the period that roots are present.
      tmp(:, :, :, :) = root_profile(:, :, :, :)
      DO jv = 1, nvm
        DO jsl = 1, nslm
          WHERE (SUM(circ_class_biomass(:, jv, :, iroot, icarbon), dim = 2) .LT. min_stomate)
            tmp(:, jv, jsl, istruc) = xios_default_val
            tmp(:, jv, jsl, ifunc) = xios_default_val
          END WHERE
        END DO
      END DO
      !CALL xios_orchidee_send_field("ROOT_PROF_STRUC",tmp(:,:,:,istruc))
        !CALL xios_orchidee_send_field("ROOT_PROF_FUNC",tmp(:,:,:,ifunc))

        ! Intermediate and bottom layers
        DO jsl = 2, nslm
        DO jv = 2, nvm
          DO ji = 1, kjpindex
            ! AD16*** Although plants can only withdraw liquid water, we compute here the water stress
              ! based on mc and the corresponding thresholds mcs, pcent, or potentially mcw and mcfc
              ! This is consistent with assuming that ice is uniformly distributed within the poral space
              ! In such a case, freezing makes mcl and the "liquid" porosity smaller than the "total" values
              ! And it is the same for all the moisture thresholds, which are proportional to porosity.
              ! Since the stress is based on relative moisture, it could thus independent from the porosity
              ! at first order, thus independent from freezing.
              ! 26-07-2017: us and humrel now based on liquid soil moisture, so the stress is stronger
              IF (new_watstress) THEN
              IF ((sm(ji, jsl) - smw(ji, jsl)) .GT. min_sechiba) THEN
                us(ji, jv, jst, jsl) = MIN(un, MAX(zero, (EXP(- alpha_watstress * ((smf(ji, jsl) - smw(ji, jsl)) / &
&(sm_nostress(ji, jsl) - smw(ji, jsl))) * ((sm_nostress(ji, jsl) - sm(ji, jsl)) / (sm(ji, jsl) - smw(ji, jsl))))))) * &
&root_profile(ji, jv, jsl, ifunc)
              ELSE
                us(ji, jv, jst, jsl) = 0.
              END IF
            ELSE
              us(ji, jv, jst, jsl) = MIN(un, MAX(zero, (sm(ji, jsl) - smw(ji, jsl)) / (sm_nostress(ji, jsl) - smw(ji, jsl)))) * &
&root_profile(ji, jv, jsl, ifunc)
            END IF
            humrelv(ji, jv, jst) = humrelv(ji, jv, jst) + us(ji, jv, jst, jsl)
          END DO
        END DO
      END DO

      !! vegstressv is the water stress for phenology in stomate
      !! It varies linearly from zero at wilting point to 1 at field capacity
      vegstressv(:, :, jst) = zero
      DO jv = 2, nvm
        DO ji = 1, kjpindex
          DO jsl = 1, nslm
            vegstressv(ji, jv, jst) = vegstressv(ji, jv, jst) + MIN(un, MAX(zero, (sm(ji, jsl) - smw(ji, jsl)) / (smf(ji, jsl) - &
&smw(ji, jsl)))) * root_profile(ji, jv, jsl, ifunc)
          END DO
        END DO
      END DO


        ! -- If the PFT is absent, the corresponding humrelv and vegstressv = 0
        DO jv = 2, nvm
        DO ji = 1, kjpindex
          IF (vegetmax_soil(ji, jv, jst) .LT. min_sechiba) THEN
            humrelv(ji, jv, jst) = zero
            vegstressv(ji, jv, jst) = zero
            us(ji, jv, jst, :) = zero
          END IF
        END DO
      END DO

        !! 6.2 We need to turn off evaporation when is_under_mcr
        !!     We set us, humrelv and vegstressv to zero in this case
        !!     WARNING: It's different from having locally us=0 in the soil layers(s) where mc<mcr
        !!              This part is crucial to preserve water conservation
        DO jsl = 1, nslm
        DO jv = 2, nvm
          WHERE (is_under_mcr(:, jst))
            us(:, jv, jst, jsl) = zero
          END WHERE
        END DO
      END DO
      DO jv = 2, nvm
        WHERE (is_under_mcr(:, jst))
          humrelv(:, jv, jst) = zero
        END WHERE
      END DO
      !rwilt and soil_wet_ns to zero in this case.
        ! They are used later for shumdiag and shumdiag_perma
        DO jsl = 1, nslm
        WHERE (is_under_mcr(:, jst))
          soil_wet_ns(:, jsl, jst) = zero
        END WHERE
      END DO

        ! Counting the nb of under_mcr occurences in each grid-cell
        WHERE (is_under_mcr(:, jst))
        undermcr = undermcr + un
      END WHERE

      !! 6.3 Calculate the volumetric soil moisture content (mc_layh and mcl_layh) needed in
      !!     thermosoil for the thermal conductivity.
      !! The multiplication by vegtot creates grid-cell average values
      ! *** To be checked for consistency with the use of nobio properties in thermosoil
      mc_layh_s = mc
      mcl_layh_s = mc
      DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mc_layh(ji, jsl) = mc_layh(ji, jsl) + mc(ji, jsl, jst) * soiltile(ji, jst) * vegtot(ji)
          mcl_layh(ji, jsl) = mcl_layh(ji, jsl) + mcl(ji, jsl, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

        !! 6.4 The hydraulic conductivities exported here are the ones used in the diffusion/redistribution
        ! (no call of hydrol_soil_coef since 2.1)
        ! We average the values of each soiltile and keep the specific value (no multiplication by vegtot)
        DO ji = 1, kjpindex
        kk_moy(ji, :) = kk_moy(ji, :) + soiltile(ji, jst) * k(ji, :)
        kk(ji, :, jst) = k(ji, :)
      END DO

        !! 6.5 We also want to export ksat at each node for CMIP6
        !  (In the output, done only once according to field_def_orchidee.xml; same averaging as for kk)
        DO jsl = 1, nslm
        ksat(:, jsl) = ksat(:, jsl) + soiltile(:, jst) * (ks(:) * kfact(jsl, :) * kfact_root(:, jsl, jst))
      END DO

      IF (printlev >= 3) WRITE(numout, *) ' prognostic/diagnostic part of hydrol_soil done for jst =', jst

    END DO
    ! end of loop on soiltile


    !! -- ENDING THE MAIN LOOP ON SOILTILES

    !! 7. Summing 3d variables into 2d variables
    CALL hydrol_diag_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, evapot, &
&vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, &
&drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)

    ! Means of wtd, runoff and drainage corrections, across soiltiles
    wtd(:) = zero
    ru_corr(:) = zero
    ru_corr2(:) = zero
    dr_corr(:) = zero
    dr_corrnum(:) = zero
    dr_force(:) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        wtd(ji) = wtd(ji) + soiltile(ji, jst) * wtd_ns(ji, jst)
        ! average over vegtot only
          IF (vegtot(ji) .GT. min_sechiba) THEN
          ! to mimic hydrol_diag_soil
          ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
          ru_corr(ji) = ru_corr(ji) + vegtot(ji) * soiltile(ji, jst) * ru_corr_ns(ji, jst)
          ru_corr2(ji) = ru_corr2(ji) + vegtot(ji) * soiltile(ji, jst) * ru_corr2_ns(ji, jst)
          dr_corr(ji) = dr_corr(ji) + vegtot(ji) * soiltile(ji, jst) * dr_corr_ns(ji, jst)
          dr_corrnum(ji) = dr_corrnum(ji) + vegtot(ji) * soiltile(ji, jst) * dr_corrnum_ns(ji, jst)
          dr_force(ji) = dr_force(ji) - vegtot(ji) * soiltile(ji, jst) * dr_force_ns(ji, jst)
          ! the sign is OK to get a negative drainage flux
        END IF
      END DO
    END DO

    ! Means local variables, including water conservation checks
    ru_infilt(:) = 0.
    qinfilt(:) = 0.
    check_infilt(:) = 0.
    check_tr(:) = 0.
    check_over(:) = 0.
    check_under(:) = 0.
    qflux(:, :) = 0.
    check_top(:) = 0.
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        IF (vegtot(ji) .GT. min_sechiba) THEN
          ! to mimic hydrol_diag_soil
          ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
          ru_infilt(ji) = ru_infilt(ji) + vegtot(ji) * soiltile(ji, jst) * ru_infilt_ns(ji, jst)
          qinfilt(ji) = qinfilt(ji) + vegtot(ji) * soiltile(ji, jst) * qinfilt_ns(ji, jst)
        END IF
      END DO
    END DO

      IF (check_cwrr) THEN
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (vegtot(ji) .GT. min_sechiba) THEN
            ! to mimic hydrol_diag_soil
            ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
            check_infilt(ji) = check_infilt(ji) + vegtot(ji) * soiltile(ji, jst) * check_infilt_ns(ji, jst)
            check_tr(ji) = check_tr(ji) + vegtot(ji) * soiltile(ji, jst) * check_tr_ns(ji, jst)
            check_over(ji) = check_over(ji) + vegtot(ji) * soiltile(ji, jst) * check_over_ns(ji, jst)
            check_under(ji) = check_under(ji) + vegtot(ji) * soiltile(ji, jst) * check_under_ns(ji, jst)
            !
            qflux(ji, :) = qflux(ji, :) + vegtot(ji) * soiltile(ji, jst) * qflux_ns(ji, :, jst)
            check_top(ji) = check_top(ji) + vegtot(ji) * soiltile(ji, jst) * check_top_ns(ji, jst)
          END IF
        END DO
      END DO
    END IF

    !! 8. COMPUTING EVAP_BARE_LIM_NS FOR NEXT TIME STEP, WITH A LOOP ON SOILTILES
    !!    The principle is to run a dummy integration of the water redistribution scheme
    !!    to check if the SM profile can sustain a potential evaporation.
    !!    If not, the dummy integration is redone from the SM profile of the end of the normal integration,
    !!    with a boundary condition leading to a very severe water limitation: mc(1)=mcr

    ! evap_bare_lim = beta factor for bare soil evaporation
    evap_bare_lim(:) = zero
    evap_bare_lim_ns(:, :) = zero

      ! Loop on soil tiles
      DO jst = 1, nstm

        !! 8.1 Save actual mc, mcl, and tmc for restoring at the end of the time step
        !!      and calculate tmcint corresponding to mc without water2infilt
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mcint(ji, jsl) = mask_soiltile(ji, jst) * mc(ji, jsl, jst)
          mclint(ji, jsl) = mask_soiltile(ji, jst) * mcl(ji, jsl, jst)
        END DO
      END DO

        DO ji = 1, kjpindex
        temp(ji) = tmc(ji, jst)
        tmcint(ji) = temp(ji) - water2infilt(ji, jst)
        ! to estimate bare soil evap based on water budget
      END DO

      !! 8.2 Since we estimate bare soile evap for the next time step, we update profil_froz_hydro and mcl
      !     (effect of mc only, the change in stempdiag is neglected)
      IF (ok_freeze_cwrr) CALL hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, jst, njsc, stempdiag)
      DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mcl(ji, jsl, jst) = MIN(mc(ji, jsl, jst), mcr(ji) + (un - profil_froz_hydro_ns(ji, jsl, jst)) * (mc(ji, jsl, jst) - &
&mcr(ji)))
          ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we keep mcl=mc
        END DO
      END DO

      !! 8.3 K and D are recomputed for the updated profile of mc/mcl
      CALL hydrol_soil_coef(mcr, mcs, kjpindex, jst, njsc)
      !! for the hydraulic architecture we need to pass the hydraulic
      !  conductivity. We save this variable in ksoil
      ksoil(:, :, jst) = k

      !! 8.4 Set the tridiagonal matrix coefficients for the diffusion/redistribution scheme
      CALL hydrol_soil_setup(kjpindex, jst)
      resolv(:) = (mask_soiltile(:, jst) .GT. 0)

        !! 8.5 We define the system of linear equations, based on matrix coefficients,

        !- Impose potential evaporation as flux_top in mm/step, assuming the water is available
        ! Note that this should lead to never have evapnu>evapot_penm(ji)

        DO ji = 1, kjpindex

          IF (vegtot(ji) .GT. min_sechiba) THEN

            ! We calculate a reduced demand, by means of a soil resistance (Sellers et al., 1992)
            ! It is based on the liquid SM only, like for us and humrel
            IF (do_rsoil) THEN
            mc_rel(ji) = tmc_litter(ji, jst) / tmcs_litter(ji)
            ! tmc_litter based on mcl
            ! based on SM in the top 4 soil layers (litter) to smooth variability
            r_soil_ns(ji, jst) = EXP(8.206 - 4.255 * mc_rel(ji))
          ELSE
            r_soil_ns(ji, jst) = zero
          END IF

          ! Aerodynamic resistance
          speed = MAX(min_wind, SQRT(u(ji) * u(ji) + v(ji) * v(ji)))
          IF (speed * tq_cdrag(ji) .GT. min_sechiba) THEN
            ra = un / (speed * tq_cdrag(ji))
            evap_soil(ji) = evapot_penm(ji) / (un + r_soil_ns(ji, jst) / ra)
          ELSE
            evap_soil(ji) = evapot_penm(ji)
          END IF

          flux_top(ji) = evap_soil(ji) * AINT(frac_bare_ns(ji, jst) + un - min_sechiba)
        ELSE

          flux_top(ji) = zero
          ! r_soil_ns needs a value to support the calculation in
          ! section "evap_bar_lim is the grid-cell scale beta"
          r_soil_ns(ji, jst) = zero

        END IF
      END DO

        ! We don't use rootsinks, because we assume there is no transpiration in the bare soil fraction (??)
        !- First layer
        DO ji = 1, kjpindex
        tmat(ji, 1, 1) = zero
        tmat(ji, 1, 2) = f(ji, 1)
        tmat(ji, 1, 3) = g1(ji, 1)
        rhs(ji, 1) = fp(ji, 1) * mcl(ji, 1, jst) + gp(ji, 1) * mcl(ji, 2, jst) - flux_top(ji) - (b(ji, 1) + b(ji, 2)) / deux * &
&(dt_sechiba / one_day)
      END DO
      !- soil body
        DO jsl = 2, nslm - 1
        DO ji = 1, kjpindex
          tmat(ji, jsl, 1) = e(ji, jsl)
          tmat(ji, jsl, 2) = f(ji, jsl)
          tmat(ji, jsl, 3) = g1(ji, jsl)
          rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + gp(ji, jsl) * mcl(ji, jsl + 1, &
&jst) + (b(ji, jsl - 1) - b(ji, jsl + 1)) * (dt_sechiba / one_day) / deux
        END DO
      END DO
      !- Last layer
        DO ji = 1, kjpindex
        jsl = nslm
        tmat(ji, jsl, 1) = e(ji, jsl)
        tmat(ji, jsl, 2) = f(ji, jsl)
        tmat(ji, jsl, 3) = zero
        rhs(ji, jsl) = ep(ji, jsl) * mcl(ji, jsl - 1, jst) + fp(ji, jsl) * mcl(ji, jsl, jst) + (b(ji, jsl - 1) + b(ji, jsl) * (un &
&- deux * free_drain_coef(ji, jst))) * (dt_sechiba / one_day) / deux
      END DO
      !- Store the equations for later use (9.6)
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          srhs(ji, jsl) = rhs(ji, jsl)
          stmat(ji, jsl, 1) = tmat(ji, jsl, 1)
          stmat(ji, jsl, 2) = tmat(ji, jsl, 2)
          stmat(ji, jsl, 3) = tmat(ji, jsl, 3)
        END DO
      END DO

      !! 8.6 Solve the diffusion equation, assuming that flux_top=evapot_penm (updates mcl)
      CALL hydrol_soil_tridiag(kjpindex, jst)

        !! 9.7 Alternative solution with mc(1)=mcr in points where the above solution leads to mcl<mcr
        ! hydrol_soil_tridiag calculates mc recursively from the top as a fonction of rhs and tmat
        ! We re-use these the above values, but for mc(1)=mcr and the related tmat

        DO ji = 1, kjpindex
        ! by construction, mc and mcl are always on the same side of mcr, so we can use mcl here
        resolv(ji) = (mcl(ji, 1, jst) .LT. (mcr(ji)) .AND. flux_top(ji) .GT. min_sechiba)
      END DO
      !! Reset the coefficient for diffusion (tridiag is only solved if resolv(ji) = .TRUE.)O
        DO jsl = 1, nslm
        !- The new condition is to put the upper layer at residual soil moisture
          DO ji = 1, kjpindex
          rhs(ji, jsl) = srhs(ji, jsl)
          tmat(ji, jsl, 1) = stmat(ji, jsl, 1)
          tmat(ji, jsl, 2) = stmat(ji, jsl, 2)
          tmat(ji, jsl, 3) = stmat(ji, jsl, 3)
        END DO
      END DO

        DO ji = 1, kjpindex
        tmat(ji, 1, 2) = un
        tmat(ji, 1, 3) = zero
        rhs(ji, 1) = mcr(ji)
      END DO

      ! Solves the diffusion equation with new surface bc where resolv=T
      CALL hydrol_soil_tridiag(kjpindex, jst)

        !! 8.8 In both case, we have drainage to be consistent with rhs
        DO ji = 1, kjpindex
        flux_bottom(ji) = mask_soiltile(ji, jst) * k(ji, nslm) * free_drain_coef(ji, jst) * (dt_sechiba / one_day)
      END DO

        !! 8.9 Water budget to assess the top flux = soil evaporation
        !      Where resolv=F at the 2nd step (9.6), it should simply be the potential evaporation

        ! Total soil moisture content for water budget

        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mc(ji, jsl, jst) = MAX(mcl(ji, jsl, jst), mcl(ji, jsl, jst) + profil_froz_hydro_ns(ji, jsl, jst) * (mc(ji, jsl, jst) - &
&mcr(ji)))
          ! if profil_froz_hydro_ns=0 (including NOT ok_freeze_cwrr) we get mc=mcl
        END DO
      END DO

        DO ji = 1, kjpindex
        tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      END DO
      DO jsl = 2, nslm - 1
        DO ji = 1, kjpindex
          tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois &
&* mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
        END DO
      END DO
      DO ji = 1, kjpindex
        tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      END DO

        ! Deduce upper flux from soil moisture variation and bottom flux
        ! TMCi-D-BSE=TMC (BSE=bare soil evap=TMCi-TMC-D)
        ! The numerical errors of tridiag close to saturation cannot be simply solved here,
        ! we can only hope they are not too large because we don't add water at this stage...
        DO ji = 1, kjpindex
        evap_bare_lim_ns(ji, jst) = mask_soiltile(ji, jst) * (tmcint(ji) - tmc(ji, jst) - flux_bottom(ji) - SUM(rootsink(ji, :, &
&jst)))
      END DO

        !! 8.10 evap_bare_lim_ns is turned from an evaporation rate to a beta
        DO ji = 1, kjpindex
        ! Here we weight evap_bare_lim_ns by the fraction of bare evaporating soil.
          ! This is given by frac_bare_ns, taking into account bare soil under vegetation
          IF (vegtot(ji) .GT. min_sechiba) THEN
          evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) * frac_bare_ns(ji, jst)
        ELSE
          evap_bare_lim_ns(ji, jst) = 0.
        END IF
      END DO

        ! We divide by evapot, which is consistent with diffuco (evap_bare_lim_ns < evapot_penm/evapot)
        ! Further decrease if tmc_litter is below the wilting point

        IF (do_rsoil) THEN
        DO ji = 1, kjpindex
          IF (evapot(ji) .GT. min_sechiba) THEN
            evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) / evapot(ji)
          ELSE
            evap_bare_lim_ns(ji, jst) = zero
            ! not redundant with the is_under_mcr case below
            ! but not necessarily useful
          END IF
          evap_bare_lim_ns(ji, jst) = MAX(MIN(evap_bare_lim_ns(ji, jst), 1.), 0.)
        END DO
      ELSE
        DO ji = 1, kjpindex
          IF ((evapot(ji) .GT. min_sechiba) .AND. (tmc_litter(ji, jst) .GT. (tmc_litter_wilt(ji, jst)))) THEN
            evap_bare_lim_ns(ji, jst) = evap_bare_lim_ns(ji, jst) / evapot(ji)
          ELSE IF ((evapot(ji) .GT. min_sechiba) .AND. (tmc_litter(ji, jst) .GT. (tmc_litter_res(ji, jst)))) THEN
            evap_bare_lim_ns(ji, jst) = (un / deux) * evap_bare_lim_ns(ji, jst) / evapot(ji)
            ! This is very arbitrary, with no justification from the literature
          ELSE
            evap_bare_lim_ns(ji, jst) = zero
          END IF
          evap_bare_lim_ns(ji, jst) = MAX(MIN(evap_bare_lim_ns(ji, jst), 1.), 0.)
        END DO
      END IF

        !! 8.11 Set evap_bare_lim_ns to zero if is_under_mcr at the end of the prognostic loop
        !!      (cf us, humrelv, vegstressv in 5.2)
        WHERE (is_under_mcr(:, jst))
        evap_bare_lim_ns(:, jst) = zero
      END WHERE

        !! 8.12 Restores mc, mcl, and tmc, to erase the effect of the dummy integrations
        !!      on these prognostic variables
        DO jsl = 1, nslm
        DO ji = 1, kjpindex
          mc(ji, jsl, jst) = mask_soiltile(ji, jst) * mcint(ji, jsl)
          mcl(ji, jsl, jst) = mask_soiltile(ji, jst) * mclint(ji, jsl)
        END DO
      END DO
      DO ji = 1, kjpindex
        tmc(ji, jst) = temp(ji)
      END DO

    END DO
    !end loop on tiles for dummy integration

      !! 9. evap_bar_lim is the grid-cell scale beta
      DO ji = 1, kjpindex
      evap_bare_lim(ji) = SUM(evap_bare_lim_ns(ji, :) * vegtot(ji) * soiltile(ji, :))
      r_soil(ji) = SUM(r_soil_ns(ji, :) * vegtot(ji) * soiltile(ji, :))
    END DO
    ! si vegtot LE min_sechiba, evap_bare_lim_ns et evap_bare_lim valent zero


    !! 10. XIOS export of local variables, including water conservation checks
    !CALL xios_orchidee_send_field("ksat",ksat) ! mm/d (for CMIP6, once)
    !CALL xios_orchidee_send_field("psi_moy",psi_moy) ! mm (for SP-MIP)
    !CALL xios_orchidee_send_field("wtd",wtd) ! in m
    !CALL xios_orchidee_send_field("ru_corr",ru_corr/dt_sechiba)   ! adjustment flux added to surface runoff (included in runoff)
    !CALL xios_orchidee_send_field("ru_corr2",ru_corr2/dt_sechiba)
    !CALL xios_orchidee_send_field("dr_corr",dr_corr/dt_sechiba)   ! adjustment flux added to drainage (included in drainage)
    !CALL xios_orchidee_send_field("dr_corrnum",dr_corrnum/dt_sechiba)
    !CALL xios_orchidee_send_field("dr_force",dr_force/dt_sechiba) ! adjustement flux added to drainage to sustain a forced wtd
    !CALL xios_orchidee_send_field("qinfilt",qinfilt/dt_sechiba)
    !CALL xios_orchidee_send_field("ru_infilt",ru_infilt/dt_sechiba)
    !CALL xios_orchidee_send_field("r_soil",r_soil) ! s/m

    !IF (check_cwrr) THEN
    !   CALL xios_orchidee_send_field("check_infilt",check_infilt/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_tr",check_tr/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_over",check_over/dt_sechiba)
    !   CALL xios_orchidee_send_field("check_under",check_under/dt_sechiba)
    !   ! Variables calculated in hydrol_diag_soil_flux
    !   CALL xios_orchidee_send_field("qflux",qflux/dt_sechiba) ! upward water flux at the low interface of each layer
    !   CALL xios_orchidee_send_field("check_top",check_top/dt_sechiba) !water budget residu in top layer
    !END IF


  END SUBROUTINE hydrol_soil
  SUBROUTINE read_dummy(altmax, avan, circ_class_biomass, e_frac, evap_bare_lim, evap_bare_lim_ns, evapot, evapot_penm, &
&F_absorption, frac_snow_nobio, humrel, irrigation, ks, mcfc, mcr, mcs, mcw, njsc, nvan, precip_rain, reinf_slope_soil, &
&reinfiltration, returnflow, snow, snowdz, soiltile, stempdiag, tot_bare_soil, tot_melt, totfrac_nobio, tq_cdrag, transpir, u, us, &
&v, veget, veget_max, vevapnu)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget
    REAL(KIND = r_std), DIMENSION(kjpindex) :: v
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us
    REAL(KIND = r_std), DIMENSION(kjpindex) :: u
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: transpir
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tq_cdrag
    REAL(KIND = r_std), DIMENSION(kjpindex) :: totfrac_nobio
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_melt
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_bare_soil
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz
    REAL(KIND = r_std), DIMENSION(kjpindex) :: snow
    REAL(KIND = r_std), DIMENSION(kjpindex) :: returnflow
    REAL(KIND = r_std), DIMENSION(kjpindex) :: reinfiltration
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: reinf_slope_soil
    REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
    REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
    REAL(KIND = r_std), DIMENSION(kjpindex) :: irrigation
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: humrel
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: F_absorption
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot_penm
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: evap_bare_lim_ns
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evap_bare_lim
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm) :: e_frac
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ncirc, nparts, nelements) :: circ_class_biomass
    REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: altmax
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil/dummy.bin', FORM = 'unformatted', STATUS = 'old')
    WRITE(*, *) '--- inside the read dummy routine for hydrol_soil ---'
    READ(1363, IOSTAT = ier) altmax
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for altmax. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) avan
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for avan. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) circ_class_biomass
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for circ_class_biomass. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) e_frac
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for e_frac. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) evap_bare_lim
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for evap_bare_lim. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) evap_bare_lim_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for evap_bare_lim_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) evapot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for evapot. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) evapot_penm
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for evapot_penm. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) F_absorption
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for F_absorption. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) frac_snow_nobio
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for frac_snow_nobio. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humrel
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humrel. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) irrigation
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for irrigation. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ks
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ks. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcfc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcfc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcs
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcs. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcw
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcw. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) njsc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for njsc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) nvan
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for nvan. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precip_rain
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precip_rain. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) reinf_slope_soil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for reinf_slope_soil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) reinfiltration
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for reinfiltration. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) returnflow
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for returnflow. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snow
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snow. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snowdz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowdz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soiltile
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soiltile. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) stempdiag
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for stempdiag. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tot_bare_soil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tot_bare_soil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tot_melt
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tot_melt. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) totfrac_nobio
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for totfrac_nobio. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tq_cdrag
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tq_cdrag. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) transpir
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for transpir. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) u
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for u. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) us
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for us. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) v
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for v. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) veget
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for veget. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) veget_max
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for veget_max. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vevapnu
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vevapnu. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
  END SUBROUTINE read_dummy
END PROGRAM main
