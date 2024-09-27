
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
  REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot
  REAL(KIND = r_std), DIMENSION(kjpindex) :: returnflow
  REAL(KIND = r_std), DIMENSION(kjpindex) :: reinfiltration
  REAL(KIND = r_std), DIMENSION(kjpindex) :: irrigation
  REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_melt
  REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
  REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
  REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
  REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
  REAL(KIND = r_std), DIMENSION(kjpindex) :: totfrac_nobio
  REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
  REAL(KIND = r_std), DIMENSION(kjpindex) :: drysoil_frac, drysoil_frac_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: runoff, runoff_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: drainage, drainage_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: shumdiag, shumdiag_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: shumdiag_perma, shumdiag_perma_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: k_litt, k_litt_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: litterhumdiag, litterhumdiag_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: humrel, humrel_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: vegstress, vegstress_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu, vevapnu_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us, us_cpu
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(veget_max, njsc, soiltile, evapot, returnflow, reinfiltration, irrigation, tot_melt, ks, nvan, avan, mcr, mcs, mcfc, mcw, precip_rain, totfrac_nobio, frac_snow_nobio, vevapnu, us, ji)
  CALL hydrol_diag_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
  ae_ns_cpu = ae_ns
  tmc_cpu = tmc
  soilmoist_cpu = soilmoist
  tmc_litt_mea_cpu = tmc_litt_mea
  soilmoist_s_cpu = soilmoist_s
  mc_cpu = mc
  profil_froz_hydro_cpu = profil_froz_hydro
  soilmoist_liquid_cpu = soilmoist_liquid
  tmc_litt_wet_mea_cpu = tmc_litt_wet_mea
  ru_ns_cpu = ru_ns
  tmc_litt_dry_mea_cpu = tmc_litt_dry_mea
  humtot_cpu = humtot
  dr_ns_cpu = dr_ns
  humrelv_cpu = humrelv
  drysoil_frac_cpu = drysoil_frac
  runoff_cpu = runoff
  drainage_cpu = drainage
  shumdiag_cpu = shumdiag
  shumdiag_perma_cpu = shumdiag_perma
  k_litt_cpu = k_litt
  litterhumdiag_cpu = litterhumdiag
  humrel_cpu = humrel
  vegstress_cpu = vegstress
  vevapnu_cpu = vevapnu
  us_cpu = us
  CALL initialization
  CALL read_dummy(veget_max, njsc, soiltile, evapot, returnflow, reinfiltration, irrigation, tot_melt, ks, nvan, avan, mcr, mcs, mcfc, mcw, precip_rain, totfrac_nobio, frac_snow_nobio, vevapnu, us, ji)
  !$ACC ENTER DATA COPYIN(ks, nvan, avan, mcr, mcs, mcfc, mcw, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_diag_soil_acc(ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(ae_ns, tmc, soilmoist, tmc_litt_mea, soilmoist_s, mc, profil_froz_hydro, soilmoist_liquid, tmc_litt_wet_mea, ru_ns, tmc_litt_dry_mea, humtot, dr_ns, humrelv, drysoil_frac, runoff, drainage, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, vevapnu, us)
  !$ACC EXIT DATA DELETE(ks, nvan, avan, mcr, mcs, mcfc, mcw, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
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
  IF (ALL(soilmoist .EQ. soilmoist_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in soilmoist_gpu are equal to soilmoist_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in soilmoist_gpu do not match soilmoist_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(soilmoist - soilmoist_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_gpu:', MINVAL(soilmoist), MAXVAL(soilmoist)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_cpu:', MINVAL(soilmoist_cpu), MAXVAL(soilmoist_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmc_litt_mea .EQ. tmc_litt_mea_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_litt_mea_gpu are equal to tmc_litt_mea_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_litt_mea_gpu do not match tmc_litt_mea_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc_litt_mea - tmc_litt_mea_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_mea_gpu:', MINVAL(tmc_litt_mea), MAXVAL(tmc_litt_mea)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_mea_cpu:', MINVAL(tmc_litt_mea_cpu), MAXVAL(tmc_litt_mea_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(soilmoist_s .EQ. soilmoist_s_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in soilmoist_s_gpu are equal to soilmoist_s_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in soilmoist_s_gpu do not match soilmoist_s_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(soilmoist_s - soilmoist_s_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_s_gpu:', MINVAL(soilmoist_s), MAXVAL(soilmoist_s)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_s_cpu:', MINVAL(soilmoist_s_cpu), MAXVAL(soilmoist_s_cpu)
    WRITE(*, *) ''
  END IF
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
  IF (ALL(profil_froz_hydro .EQ. profil_froz_hydro_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in profil_froz_hydro_gpu are equal to profil_froz_hydro_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in profil_froz_hydro_gpu do not match profil_froz_hydro_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(profil_froz_hydro - profil_froz_hydro_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_gpu:', MINVAL(profil_froz_hydro), MAXVAL(profil_froz_hydro)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_cpu:', MINVAL(profil_froz_hydro_cpu), MAXVAL(profil_froz_hydro_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(soilmoist_liquid .EQ. soilmoist_liquid_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in soilmoist_liquid_gpu are equal to soilmoist_liquid_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in soilmoist_liquid_gpu do not match soilmoist_liquid_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(soilmoist_liquid - soilmoist_liquid_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_liquid_gpu:', MINVAL(soilmoist_liquid), MAXVAL(soilmoist_liquid)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of soilmoist_liquid_cpu:', MINVAL(soilmoist_liquid_cpu), MAXVAL(soilmoist_liquid_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmc_litt_wet_mea .EQ. tmc_litt_wet_mea_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_litt_wet_mea_gpu are equal to tmc_litt_wet_mea_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_litt_wet_mea_gpu do not match tmc_litt_wet_mea_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc_litt_wet_mea - tmc_litt_wet_mea_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_wet_mea_gpu:', MINVAL(tmc_litt_wet_mea), MAXVAL(tmc_litt_wet_mea)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_wet_mea_cpu:', MINVAL(tmc_litt_wet_mea_cpu), MAXVAL(tmc_litt_wet_mea_cpu)
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
  IF (ALL(tmc_litt_dry_mea .EQ. tmc_litt_dry_mea_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_litt_dry_mea_gpu are equal to tmc_litt_dry_mea_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_litt_dry_mea_gpu do not match tmc_litt_dry_mea_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc_litt_dry_mea - tmc_litt_dry_mea_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_dry_mea_gpu:', MINVAL(tmc_litt_dry_mea), MAXVAL(tmc_litt_dry_mea)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_litt_dry_mea_cpu:', MINVAL(tmc_litt_dry_mea_cpu), MAXVAL(tmc_litt_dry_mea_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(humtot .EQ. humtot_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in humtot_gpu are equal to humtot_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in humtot_gpu do not match humtot_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(humtot - humtot_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humtot_gpu:', MINVAL(humtot), MAXVAL(humtot)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humtot_cpu:', MINVAL(humtot_cpu), MAXVAL(humtot_cpu)
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
  IF (ALL(drysoil_frac .EQ. drysoil_frac_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in drysoil_frac_gpu are equal to drysoil_frac_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in drysoil_frac_gpu do not match drysoil_frac_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(drysoil_frac - drysoil_frac_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drysoil_frac_gpu:', MINVAL(drysoil_frac), MAXVAL(drysoil_frac)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drysoil_frac_cpu:', MINVAL(drysoil_frac_cpu), MAXVAL(drysoil_frac_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(runoff .EQ. runoff_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in runoff_gpu are equal to runoff_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in runoff_gpu do not match runoff_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(runoff - runoff_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of runoff_gpu:', MINVAL(runoff), MAXVAL(runoff)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of runoff_cpu:', MINVAL(runoff_cpu), MAXVAL(runoff_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(drainage .EQ. drainage_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in drainage_gpu are equal to drainage_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in drainage_gpu do not match drainage_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(drainage - drainage_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drainage_gpu:', MINVAL(drainage), MAXVAL(drainage)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drainage_cpu:', MINVAL(drainage_cpu), MAXVAL(drainage_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(shumdiag .EQ. shumdiag_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in shumdiag_gpu are equal to shumdiag_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in shumdiag_gpu do not match shumdiag_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(shumdiag - shumdiag_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of shumdiag_gpu:', MINVAL(shumdiag), MAXVAL(shumdiag)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of shumdiag_cpu:', MINVAL(shumdiag_cpu), MAXVAL(shumdiag_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(shumdiag_perma .EQ. shumdiag_perma_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in shumdiag_perma_gpu are equal to shumdiag_perma_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in shumdiag_perma_gpu do not match shumdiag_perma_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(shumdiag_perma - shumdiag_perma_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of shumdiag_perma_gpu:', MINVAL(shumdiag_perma), MAXVAL(shumdiag_perma)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of shumdiag_perma_cpu:', MINVAL(shumdiag_perma_cpu), MAXVAL(shumdiag_perma_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(k_litt .EQ. k_litt_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in k_litt_gpu are equal to k_litt_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in k_litt_gpu do not match k_litt_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(k_litt - k_litt_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of k_litt_gpu:', MINVAL(k_litt), MAXVAL(k_litt)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of k_litt_cpu:', MINVAL(k_litt_cpu), MAXVAL(k_litt_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(litterhumdiag .EQ. litterhumdiag_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in litterhumdiag_gpu are equal to litterhumdiag_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in litterhumdiag_gpu do not match litterhumdiag_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(litterhumdiag - litterhumdiag_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of litterhumdiag_gpu:', MINVAL(litterhumdiag), MAXVAL(litterhumdiag)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of litterhumdiag_cpu:', MINVAL(litterhumdiag_cpu), MAXVAL(litterhumdiag_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(humrel .EQ. humrel_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in humrel_gpu are equal to humrel_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in humrel_gpu do not match humrel_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(humrel - humrel_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humrel_gpu:', MINVAL(humrel), MAXVAL(humrel)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humrel_cpu:', MINVAL(humrel_cpu), MAXVAL(humrel_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(vegstress .EQ. vegstress_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in vegstress_gpu are equal to vegstress_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in vegstress_gpu do not match vegstress_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(vegstress - vegstress_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vegstress_gpu:', MINVAL(vegstress), MAXVAL(vegstress)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vegstress_cpu:', MINVAL(vegstress_cpu), MAXVAL(vegstress_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(vevapnu .EQ. vevapnu_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in vevapnu_gpu are equal to vevapnu_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in vevapnu_gpu do not match vevapnu_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(vevapnu - vevapnu_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vevapnu_gpu:', MINVAL(vevapnu), MAXVAL(vevapnu)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of vevapnu_cpu:', MINVAL(vevapnu_cpu), MAXVAL(vevapnu_cpu)
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
  CONTAINS


  !! ================================================================================================================================
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
  !_ ================================================================================================================================
  !_ hydrol_diag_soil

  SUBROUTINE hydrol_diag_soil_acc(ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
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

    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: i
    REAL(KIND = r_std) :: mask_vegtot
    REAL(KIND = r_std) :: k_tmp
    REAL(KIND = r_std) :: tmc_litter_ratio

    !_ ================================================================================================================================
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
      runoff(ji) = mask_vegtot * (runoff(ji) + vegtot(ji) * soiltile(ji, jst) * ru_ns(ji, jst)) + (1 - mask_vegtot) * (tot_melt(ji) + irrigation(ji) + returnflow(ji) + reinfiltration(ji))
      humtot(ji) = mask_vegtot * (humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst))
      IF (ok_freeze_cwrr) THEN
        !  profil_froz_hydro_ns comes from hydrol_soil, to remain the same as in the prognotic loop
        profil_froz_hydro(ji, :) = mask_vegtot * (profil_froz_hydro(ji, :) + vegtot(ji) * soiltile(ji, jst) * profil_froz_hydro_ns(ji, :, jst))
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
      drysoil_frac(ji) = un + MAX(MIN((tmc_litt_dry_mea(ji) - tmc_litt_mea(ji)) / (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji)), zero), - un)
    ELSE
      drysoil_frac(ji) = zero
    END IF

    ! Calculate soilmoist, as a function of total water content (mc)
    ! We average the values of each soiltile and multiply by vegtot to transform to a grid-cell mean
    soilmoist(ji, :) = zero
    DO jst = 1, nstm
      soilmoist(ji, 1) = soilmoist(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        soilmoist(ji, jsl) = soilmoist(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist(ji, nslm) = soilmoist(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
    END DO
    soilmoist(ji, :) = soilmoist(ji, :) * vegtot(ji)
    ! conversion to grid-cell average

    soilmoist_s(ji, :, :) = zero
    DO jst = 1, nstm
      soilmoist_s(ji, 1, nstm) = soilmoist_s(ji, 1, nstm) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        soilmoist_s(ji, jsl, nstm) = soilmoist_s(ji, jsl, nstm) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist_s(ji, nslm, nstm) = soilmoist_s(ji, nslm, nstm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
    END DO
    soilmoist_s(ji, :, :) = soilmoist_s(ji, :, :) * vegtot(ji)
    ! conversion to grid-cell average

    soilmoist_liquid(ji, :) = zero
    DO jst = 1, nstm
      soilmoist_liquid(ji, 1) = soilmoist_liquid(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        soilmoist_liquid(ji, jsl) = soilmoist_liquid(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit)
      END DO
      soilmoist_liquid(ji, nslm) = soilmoist_liquid(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mcl(ji, nslm, jst) + mcl(ji, nslm - 1, jst)) / huit
    END DO
    soilmoist_liquid(ji, :) = soilmoist_liquid(ji, :) * vegtot_old(ji)
    ! grid cell average


      ! Shumdiag: we start from soil_wet_ns, change the range over which the relative moisture is calculated,
      ! then do a spatial average, excluding the nobio fraction on which stomate doesn't act
      DO jst = 1, nstm
      DO jsl = 1, nslm
        shumdiag(ji, jsl) = shumdiag(ji, jsl) + soil_wet_ns(ji, jsl, jst) * soiltile(ji, jst) * ((mcs(ji) - mcw(ji)) / (mcfc(ji) - mcw(ji)))
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


    !! ================================================================================================================================
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
    !_ ================================================================================================================================
    !_ hydrol_diag_soil

    SUBROUTINE hydrol_diag_soil(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, veget_max, soiltile, njsc, runoff, drainage, evapot, vevapnu, returnflow, reinfiltration, irrigation, shumdiag, shumdiag_perma, k_litt, litterhumdiag, humrel, vegstress, drysoil_frac, tot_melt, us, precip_rain, totfrac_nobio, frac_snow_nobio)
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

    INTEGER(KIND = i_std) :: ji, jv, jsl, jst, i
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mask_vegtot
    REAL(KIND = r_std) :: k_tmp, tmc_litter_ratio

    !_ ================================================================================================================================
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
        runoff(ji) = mask_vegtot(ji) * (runoff(ji) + vegtot(ji) * soiltile(ji, jst) * ru_ns(ji, jst)) + (1 - mask_vegtot(ji)) * (tot_melt(ji) + irrigation(ji) + returnflow(ji) + reinfiltration(ji))
        humtot(ji) = mask_vegtot(ji) * (humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst))
        IF (ok_freeze_cwrr) THEN
          !  profil_froz_hydro_ns comes from hydrol_soil, to remain the same as in the prognotic loop
          profil_froz_hydro(ji, :) = mask_vegtot(ji) * (profil_froz_hydro(ji, :) + vegtot(ji) * soiltile(ji, jst) * profil_froz_hydro_ns(ji, :, jst))
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
        drysoil_frac(ji) = un + MAX(MIN((tmc_litt_dry_mea(ji) - tmc_litt_mea(ji)) / (tmc_litt_wet_mea(ji) - tmc_litt_dry_mea(ji)), zero), - un)
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
          soilmoist(ji, jsl) = soilmoist(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist(ji, nslm) = soilmoist(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      soilmoist(ji, :) = soilmoist(ji, :) * vegtot(ji)
      ! conversion to grid-cell average
    END DO

    soilmoist_s(:, :, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        soilmoist_s(ji, 1, nstm) = soilmoist_s(ji, 1, nstm) + soiltile(ji, jst) * dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
        DO jsl = 2, nslm - 1
          soilmoist_s(ji, jsl, nstm) = soilmoist_s(ji, jsl, nstm) + soiltile(ji, jst) * (dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist_s(ji, nslm, nstm) = soilmoist_s(ji, nslm, nstm) + soiltile(ji, jst) * dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      soilmoist_s(ji, :, :) = soilmoist_s(ji, :, :) * vegtot(ji)
      ! conversion to grid-cell average
    END DO

    soilmoist_liquid(:, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        soilmoist_liquid(ji, 1) = soilmoist_liquid(ji, 1) + soiltile(ji, jst) * dz(2) * (trois * mcl(ji, 1, jst) + mcl(ji, 2, jst)) / huit
        DO jsl = 2, nslm - 1
          soilmoist_liquid(ji, jsl) = soilmoist_liquid(ji, jsl) + soiltile(ji, jst) * (dz(jsl) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcl(ji, jsl, jst) + mcl(ji, jsl + 1, jst)) / huit)
        END DO
        soilmoist_liquid(ji, nslm) = soilmoist_liquid(ji, nslm) + soiltile(ji, jst) * dz(nslm) * (trois * mcl(ji, nslm, jst) + mcl(ji, nslm - 1, jst)) / huit
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
          shumdiag(ji, jsl) = shumdiag(ji, jsl) + soil_wet_ns(ji, jsl, jst) * soiltile(ji, jst) * ((mcs(ji) - mcw(ji)) / (mcfc(ji) - mcw(ji)))
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
  SUBROUTINE read_dummy(veget_max, njsc, soiltile, evapot, returnflow, reinfiltration, irrigation, tot_melt, ks, nvan, avan, mcr, mcs, mcfc, mcw, precip_rain, totfrac_nobio, frac_snow_nobio, vevapnu, us, ji)
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us
    REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
    REAL(KIND = r_std), DIMENSION(kjpindex) :: totfrac_nobio
    REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
    REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_melt
    REAL(KIND = r_std), DIMENSION(kjpindex) :: irrigation
    REAL(KIND = r_std), DIMENSION(kjpindex) :: reinfiltration
    REAL(KIND = r_std), DIMENSION(kjpindex) :: returnflow
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evapot
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    CALL random_number(veget_max)
    njsc = 2
    CALL random_number(soiltile)
    CALL random_number(evapot)
    CALL random_number(returnflow)
    CALL random_number(reinfiltration)
    CALL random_number(irrigation)
    CALL random_number(tot_melt)
    CALL random_number(ks)
    CALL random_number(nvan)
    CALL random_number(avan)
    CALL random_number(mcr)
    CALL random_number(mcs)
    CALL random_number(mcfc)
    CALL random_number(mcw)
    CALL random_number(precip_rain)
    CALL random_number(totfrac_nobio)
    CALL random_number(frac_snow_nobio)
    CALL random_number(vevapnu)
    CALL random_number(us)
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
