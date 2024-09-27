
MODULE module_global
  USE xios_orchidee, ONLY: xios_orchidee_send_field
  USE ioipsl_para, ONLY: ipslerr_p
  IMPLICIT NONE
  INTEGER, PARAMETER :: i_std = 4
  INTEGER, PARAMETER :: r_std = 8
  INTEGER(KIND = i_std), PARAMETER :: nslm = 11
  INTEGER(KIND = i_std), PARAMETER :: nvm = 13
  INTEGER(KIND = i_std), PARAMETER :: nstm = 3
  INTEGER(KIND = i_std), PARAMETER :: kjpindex = 4716
  INTEGER :: ier
  INTEGER :: seed(64) = 1
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: mille = 1000._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: w_time = 1.0_r_std
  INTEGER(KIND = i_std), PARAMETER :: iice = 1
  INTEGER(KIND = i_std), PARAMETER :: nnobio = 1
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  REAL(KIND = r_std), PARAMETER :: allowed_err = 2.0E-8_r_std
  REAL(KIND = r_std), PARAMETER :: kilo_to_unit = 1.0E03
  REAL(KIND = r_std), PARAMETER :: ZeroCelsius = 273.15
  INTEGER(KIND = i_std), PARAMETER :: ndepths = 2
  INTEGER(KIND = i_std), PARAMETER :: istruc = 1
  INTEGER(KIND = i_std), PARAMETER :: nroot_prof = 2
  INTEGER(KIND = i_std), PARAMETER :: inode = 1
  INTEGER(KIND = i_std), PARAMETER :: iinterface = 2
  INTEGER(KIND = i_std), PARAMETER :: ifunc = 2
  INTEGER(KIND = i_std), PARAMETER :: nelements = 2
  INTEGER(KIND = i_std), PARAMETER :: iroot = 6
  INTEGER(KIND = i_std), PARAMETER :: nparts = 9
  REAL(KIND = r_std), PARAMETER :: quatre = 4._r_std
  REAL(KIND = r_std), PARAMETER :: undef_sechiba = 1.E+20_r_std
  INTEGER(KIND = i_std), PARAMETER :: icarbon = 1
  REAL(KIND = r_std), PARAMETER :: min_stomate = 1.E-8_r_std
  INTEGER(KIND = i_std), PARAMETER :: nscm_usda = 13
  LOGICAL :: check_cwrr
  REAL(KIND = r_std) :: alpha_nudge_mc
  REAL(KIND = r_std) :: zmaxh
  REAL(KIND = r_std) :: one_day
  REAL(KIND = r_std) :: dt_sechiba
  INTEGER :: imax
  LOGICAL :: ok_freeze_cwrr
  INTEGER(KIND = i_std) :: numout = 6
  LOGICAL :: is_tuzet_hydrol_arch = .FALSE.
  LOGICAL :: ok_hydrol_arch
  INTEGER(KIND = i_std) :: plev = 0
  INTEGER(KIND = i_std) :: err_act = 1
  REAL(KIND = r_std) :: maxaltmax = 2.
  LOGICAL :: doponds = .FALSE.
  LOGICAL :: old_irrig_scheme = .FALSE.
  REAL(KIND = r_std) :: lai_irrig_min = 0.1
  REAL(KIND = r_std) :: beta_irrig = 0.9
  LOGICAL :: ok_nudge_mc
  LOGICAL :: new_watstress = .FALSE.
  INTEGER(KIND = i_std) :: ncirc = 1
  INTEGER(KIND = i_std) :: nsnow
  LOGICAL :: irrigated_soiltile = .FALSE.
  REAL(KIND = r_std) :: alpha_watstress = 1.
  REAL(KIND = r_std) :: min_wind = 0.1
  INTEGER :: printlev = 2
  INTEGER :: dim
  LOGICAL :: do_rsoil = .TRUE.
  REAL(KIND = r_std) :: xios_default_val = 0
  INTEGER :: irrig_st = 3
  LOGICAL :: kfact_root_const
  INTEGER(KIND = i_std) :: nscm = nscm_usda
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc_read_current
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_aux
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_soiltile
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: fp
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: free_drain_coef
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: g1
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: f
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ep
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: e
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: gp
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc, tmc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns, profil_froz_hydro_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soilmoist_s
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_awet
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: profil_froz_hydro
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl, mcl_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist_liquid
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_wet_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ru_ns, ru_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_adry
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soil_wet_ns, soil_wet_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter, tmc_litter_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_res
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_dry_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: dr_ns, dr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_sat
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegstressv, vegstressv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: frac_bare_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv, humrelv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soil_wet_litter, soil_wet_litter_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: subsinksoil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: qflux_ns, qflux_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: rootsink
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: check_top_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tr_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegetmax_soil
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: resolv, resolv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: tmat, tmat_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: rhs, rhs_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root, kfact_root_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kfact
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: a_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: b_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: d_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: b
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: zdr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: znh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: max_root_depth
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humcste
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: ext_coeff_vegetfrac
  REAL(KIND = r_std), PARAMETER, DIMENSION(nscm_usda) :: ks_usda = (/7128.0_r_std, 3501.6_r_std, 1060.8_r_std, 108.0_r_std, 60.0_r_std, 249.6_r_std, 314.4_r_std, 16.8_r_std, 62.4_r_std, 28.8_r_std, 4.8_r_std, 48.0_r_std, 6131.4_r_std/)
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: avan_mod_tab
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: undermcr, undermcr_cpu
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: natural
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kk, kk_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: stmat, stmat_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: pcent
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: water2infilt, water2infilt_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_field
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: srhs, srhs_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: nvan_mod_tab
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: zz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: zwt_force
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: nslm_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_wilt
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kk_moy, kk_moy_cpu
  !$ACC DECLARE COPYIN(zero, trois, huit, mille, min_sechiba, deux, un, w_time, iice, nnobio, imin, numout, allowed_err, is_tuzet_hydrol_arch, kilo_to_unit, ZeroCelsius, plev, ndepths, istruc, nroot_prof, inode, err_act, iinterface, ifunc, maxaltmax, nelements, doponds, ks_usda, old_irrig_scheme, lai_irrig_min, beta_irrig, new_watstress, iroot, ncirc, nparts, irrigated_soiltile, alpha_watstress, quatre, min_wind, undef_sechiba, printlev, do_rsoil, xios_default_val, icarbon, irrig_st, min_stomate, nscm_usda, nscm)
  !$ACC DECLARE CREATE(mc, dz, check_cwrr, mc_read_current, tmc_aux, alpha_nudge_mc, mask_soiltile, zmaxh, one_day, fp, dt_sechiba, free_drain_coef, d, g1, f, ep, a, e, gp, ae_ns, tmc, profil_froz_hydro_ns, imax, soilmoist, tmc_litt_mea, soilmoist_s, vegtot, k_lin, tmc_litter_awet, profil_froz_hydro, mcl, dh, soilmoist_liquid, tmc_litt_wet_mea, ru_ns, vegtot_old, tmc_litter_adry, soil_wet_ns, tmc_litter, ok_freeze_cwrr, tmc_litter_res, tmc_litt_dry_mea, humtot, dr_ns, tmc_litter_sat, vegstressv, frac_bare_ns, humrelv, soil_wet_litter, subsinksoil, qflux_ns, rootsink, check_top_ns, precisol, tr_ns, precisol_ns, ok_hydrol_arch, vegetmax_soil, pref_soil_veg, resolv, tmat, rhs, kfact_root, k, kfact, a_lin, b_lin, d_lin, b, zdr, znh, max_root_depth, humcste, ext_coeff_vegetfrac, avan_mod_tab, undermcr, natural, kk, ok_nudge_mc, stmat, pcent, water2infilt, tmc_litter_field, nsnow, srhs, dim, nvan_mod_tab, zz, zwt_force, nslm_root, tmc_litter_wilt, kfact_root_const, kk_moy)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    check_cwrr = .TRUE.
    CALL random_number(alpha_nudge_mc)
    CALL random_number(zmaxh)
    CALL random_number(one_day)
    CALL random_number(dt_sechiba)
    imax = 2
    ok_freeze_cwrr = .TRUE.
    ok_hydrol_arch = .TRUE.
    ok_nudge_mc = .TRUE.
    nsnow = 2
    dim = 2
    kfact_root_const = .TRUE.
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(mc_read_current(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmc_aux(kjpindex, nstm), STAT = ier)
    ALLOCATE(mask_soiltile(kjpindex, nstm), STAT = ier)
    ALLOCATE(fp(kjpindex, nslm), STAT = ier)
    ALLOCATE(free_drain_coef(kjpindex, nstm), STAT = ier)
    ALLOCATE(d(kjpindex, nslm), STAT = ier)
    ALLOCATE(g1(kjpindex, nslm), STAT = ier)
    ALLOCATE(f(kjpindex, nslm), STAT = ier)
    ALLOCATE(ep(kjpindex, nslm), STAT = ier)
    ALLOCATE(a(kjpindex, nslm), STAT = ier)
    ALLOCATE(e(kjpindex, nslm), STAT = ier)
    ALLOCATE(gp(kjpindex, nslm), STAT = ier)
    ALLOCATE(ae_ns(kjpindex, nstm), ae_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc(kjpindex, nstm), tmc_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), profil_froz_hydro_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(soilmoist(kjpindex, nslm), STAT = ier)
    ALLOCATE(tmc_litt_mea(kjpindex), STAT = ier)
    ALLOCATE(soilmoist_s(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(vegtot(kjpindex), STAT = ier)
    ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(tmc_litter_awet(kjpindex, nstm), STAT = ier)
    ALLOCATE(profil_froz_hydro(kjpindex, nslm), STAT = ier)
    ALLOCATE(mcl(kjpindex, nslm, nstm), mcl_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(dh(nslm), STAT = ier)
    ALLOCATE(soilmoist_liquid(kjpindex, nslm), STAT = ier)
    ALLOCATE(tmc_litt_wet_mea(kjpindex), STAT = ier)
    ALLOCATE(ru_ns(kjpindex, nstm), ru_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    ALLOCATE(tmc_litter_adry(kjpindex, nstm), STAT = ier)
    ALLOCATE(soil_wet_ns(kjpindex, nslm, nstm), soil_wet_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmc_litter(kjpindex, nstm), tmc_litter_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litter_res(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litt_dry_mea(kjpindex), STAT = ier)
    ALLOCATE(humtot(kjpindex), STAT = ier)
    ALLOCATE(dr_ns(kjpindex, nstm), dr_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litter_sat(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegstressv(kjpindex, nvm, nstm), vegstressv_cpu(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(frac_bare_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(humrelv(kjpindex, nvm, nstm), humrelv_cpu(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(soil_wet_litter(kjpindex, nstm), soil_wet_litter_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(subsinksoil(kjpindex), STAT = ier)
    ALLOCATE(qflux_ns(kjpindex, nslm, nstm), qflux_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(rootsink(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(check_top_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(precisol(kjpindex, nvm), STAT = ier)
    ALLOCATE(tr_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(precisol_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegetmax_soil(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(pref_soil_veg(nvm), STAT = ier)
    ALLOCATE(resolv(kjpindex), resolv_cpu(kjpindex), STAT = ier)
    ALLOCATE(tmat(kjpindex, nslm, 3), tmat_cpu(kjpindex, nslm, 3), STAT = ier)
    ALLOCATE(rhs(kjpindex, nslm), rhs_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(kfact_root(kjpindex, nslm, nstm), kfact_root_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(k(kjpindex, nslm), STAT = ier)
    ALLOCATE(kfact(nslm, kjpindex), STAT = ier)
    ALLOCATE(a_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(b_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(d_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(b(kjpindex, nslm), STAT = ier)
    ALLOCATE(zdr(0 : nslm), STAT = ier)
    ALLOCATE(znh(nslm), STAT = ier)
    ALLOCATE(max_root_depth(nvm), STAT = ier)
    ALLOCATE(humcste(nvm), STAT = ier)
    ALLOCATE(ext_coeff_vegetfrac(nvm), STAT = ier)
    ALLOCATE(avan_mod_tab(nslm, kjpindex), STAT = ier)
    ALLOCATE(undermcr(kjpindex), undermcr_cpu(kjpindex), STAT = ier)
    ALLOCATE(natural(nvm), STAT = ier)
    ALLOCATE(kk(kjpindex, nslm, nstm), kk_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(stmat(kjpindex, nslm, 3), stmat_cpu(kjpindex, nslm, 3), STAT = ier)
    ALLOCATE(pcent(nscm), STAT = ier)
    ALLOCATE(water2infilt(kjpindex, nstm), water2infilt_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litter_field(kjpindex, nstm), STAT = ier)
    ALLOCATE(srhs(kjpindex, nslm), srhs_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(nvan_mod_tab(nslm, kjpindex), STAT = ier)
    ALLOCATE(zz(nslm), STAT = ier)
    ALLOCATE(zwt_force(kjpindex, nstm), STAT = ier)
    ALLOCATE(nslm_root(kjpindex), STAT = ier)
    ALLOCATE(tmc_litter_wilt(kjpindex, nstm), STAT = ier)
    ALLOCATE(kk_moy(kjpindex, nslm), kk_moy_cpu(kjpindex, nslm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(mc)
    CALL random_number(dz)
    CALL random_number(mc_read_current)
    CALL random_number(tmc_aux)
    mask_soiltile = 2
    CALL random_number(fp)
    CALL random_number(free_drain_coef)
    CALL random_number(d)
    CALL random_number(g1)
    CALL random_number(f)
    CALL random_number(ep)
    CALL random_number(a)
    CALL random_number(e)
    CALL random_number(gp)
    CALL random_number(ae_ns)
    CALL random_number(tmc)
    CALL random_number(profil_froz_hydro_ns)
    CALL random_number(soilmoist)
    CALL random_number(tmc_litt_mea)
    CALL random_number(soilmoist_s)
    CALL random_number(vegtot)
    CALL random_number(k_lin)
    CALL random_number(tmc_litter_awet)
    CALL random_number(profil_froz_hydro)
    CALL random_number(mcl)
    CALL random_number(dh)
    CALL random_number(soilmoist_liquid)
    CALL random_number(tmc_litt_wet_mea)
    CALL random_number(ru_ns)
    CALL random_number(vegtot_old)
    CALL random_number(tmc_litter_adry)
    CALL random_number(soil_wet_ns)
    CALL random_number(tmc_litter)
    CALL random_number(tmc_litter_res)
    CALL random_number(tmc_litt_dry_mea)
    CALL random_number(humtot)
    CALL random_number(dr_ns)
    CALL random_number(tmc_litter_sat)
    CALL random_number(vegstressv)
    CALL random_number(frac_bare_ns)
    CALL random_number(humrelv)
    CALL random_number(soil_wet_litter)
    CALL random_number(subsinksoil)
    CALL random_number(qflux_ns)
    CALL random_number(rootsink)
    CALL random_number(check_top_ns)
    CALL random_number(precisol)
    CALL random_number(tr_ns)
    CALL random_number(precisol_ns)
    CALL random_number(vegetmax_soil)
    pref_soil_veg = 2
    resolv = .TRUE.
    CALL random_number(tmat)
    CALL random_number(rhs)
    CALL random_number(kfact_root)
    CALL random_number(k)
    CALL random_number(kfact)
    CALL random_number(a_lin)
    CALL random_number(b_lin)
    CALL random_number(d_lin)
    CALL random_number(b)
    CALL random_number(zdr)
    CALL random_number(znh)
    CALL random_number(max_root_depth)
    CALL random_number(humcste)
    CALL random_number(ext_coeff_vegetfrac)
    CALL random_number(avan_mod_tab)
    CALL random_number(undermcr)
    natural = .TRUE.
    CALL random_number(kk)
    CALL random_number(stmat)
    CALL random_number(pcent)
    CALL random_number(water2infilt)
    CALL random_number(tmc_litter_field)
    CALL random_number(srhs)
    CALL random_number(nvan_mod_tab)
    CALL random_number(zz)
    CALL random_number(zwt_force)
    nslm_root = 2
    CALL random_number(tmc_litter_wilt)
    CALL random_number(kk_moy)
    !$ACC UPDATE DEVICE(mc, dz, check_cwrr, mc_read_current, tmc_aux, alpha_nudge_mc, mask_soiltile, zmaxh, one_day, fp, dt_sechiba, free_drain_coef, d, g1, f, ep, a, e, gp, ae_ns, tmc, profil_froz_hydro_ns, imax, soilmoist, tmc_litt_mea, soilmoist_s, vegtot, k_lin, tmc_litter_awet, profil_froz_hydro, mcl, dh, soilmoist_liquid, tmc_litt_wet_mea, ru_ns, vegtot_old, tmc_litter_adry, soil_wet_ns, tmc_litter, ok_freeze_cwrr, tmc_litter_res, tmc_litt_dry_mea, humtot, dr_ns, tmc_litter_sat, vegstressv, frac_bare_ns, humrelv, soil_wet_litter, subsinksoil, qflux_ns, rootsink, check_top_ns, precisol, tr_ns, precisol_ns, ok_hydrol_arch, vegetmax_soil, pref_soil_veg, resolv, tmat, rhs, kfact_root, k, kfact, a_lin, b_lin, d_lin, b, zdr, znh, max_root_depth, humcste, ext_coeff_vegetfrac, avan_mod_tab, undermcr, natural, kk, ok_nudge_mc, stmat, pcent, water2infilt, tmc_litter_field, nsnow, srhs, dim, nvan_mod_tab, zz, zwt_force, nslm_root, tmc_litter_wilt, kfact_root_const, kk_moy)
  END SUBROUTINE initialization
END MODULE module_global
