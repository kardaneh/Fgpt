
MODULE module_global
  USE ioipsl_para, ONLY: ipslerr_p
  IMPLICIT NONE
  INTEGER, PARAMETER :: i_std = 4
  INTEGER, PARAMETER :: r_std = 8
  INTEGER(KIND = i_std), PARAMETER :: nsnow = 3
  INTEGER(KIND = i_std), PARAMETER :: nslm = 11
  INTEGER(KIND = i_std), PARAMETER :: nvm = 15
  INTEGER(KIND = i_std), PARAMETER :: nstm = 3
  INTEGER(KIND = i_std), PARAMETER :: kjpindex = 4717
  INTEGER :: ier
  INTEGER(KIND = i_std) :: ic0, ic
  REAL(KIND = r_std) :: icr, start_time, stop_time
  REAL(KIND = r_std), PARAMETER :: ZeroCelsius = 273.15
  REAL(KIND = r_std), PARAMETER :: allowed_err = 2.0E-8_r_std
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  INTEGER(KIND = i_std), PARAMETER :: icarbon = 1
  INTEGER(KIND = i_std), PARAMETER :: ifunc = 2
  INTEGER(KIND = i_std), PARAMETER :: iice = 1
  INTEGER(KIND = i_std), PARAMETER :: iinterface = 2
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  INTEGER(KIND = i_std), PARAMETER :: inode = 1
  INTEGER(KIND = i_std), PARAMETER :: iroot = 6
  INTEGER(KIND = i_std), PARAMETER :: istruc = 1
  REAL(KIND = r_std), PARAMETER :: kilo_to_unit = 1.0E03
  REAL(KIND = r_std), PARAMETER :: lhf = 0.3336 * 1.E6
  REAL(KIND = r_std), PARAMETER :: mille = 1000._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: min_stomate = 1.E-8_r_std
  INTEGER(KIND = i_std), PARAMETER :: ndepths = 2
  INTEGER(KIND = i_std), PARAMETER :: nelements = 2
  INTEGER(KIND = i_std), PARAMETER :: nnobio = 1
  INTEGER(KIND = i_std), PARAMETER :: nparts = 9
  INTEGER(KIND = i_std), PARAMETER :: nroot_prof = 2
  INTEGER(KIND = i_std), PARAMETER :: nscm_usda = 13
  REAL(KIND = r_std), PARAMETER :: quatre = 4._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: undef_sechiba = 1.E+20_r_std
  REAL(KIND = r_std), PARAMETER :: w_time = 1.0_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std) :: alpha_nudge_mc
  REAL(KIND = r_std) :: alpha_watstress = 1.
  REAL(KIND = r_std) :: beta_irrig = 0.9
  LOGICAL :: check_cwrr
  LOGICAL :: do_rsoil = .TRUE.
  LOGICAL :: doponds = .FALSE.
  REAL(KIND = r_std) :: dt_sechiba
  INTEGER(KIND = i_std) :: err_act = 1
  REAL(KIND = r_std) :: fr_center
  REAL(KIND = r_std) :: fr_dT
  REAL(KIND = r_std) :: froz_frac_corr
  INTEGER :: imax
  INTEGER :: irrig_st = 3
  LOGICAL :: irrigated_soiltile = .FALSE.
  LOGICAL :: is_tuzet_hydrol_arch = .FALSE.
  LOGICAL :: kfact_root_const
  REAL(KIND = r_std) :: lai_irrig_min = 0.1
  REAL(KIND = r_std) :: max_froz_hydro
  REAL(KIND = r_std) :: maxaltmax = 2.
  REAL(KIND = r_std) :: min_wind = 0.1
  INTEGER(KIND = i_std) :: ncirc = 1
  LOGICAL :: new_watstress = .FALSE.
  INTEGER(KIND = i_std) :: nscm = nscm_usda
  INTEGER(KIND = i_std) :: numout = 6
  LOGICAL :: ok_freeze_cwrr
  LOGICAL :: ok_hydrol_arch
  LOGICAL :: ok_nudge_mc
  LOGICAL :: ok_thermodynamical_freezing
  LOGICAL :: old_irrig_scheme = .FALSE.
  REAL(KIND = r_std) :: one_day
  INTEGER(KIND = i_std) :: plev = 0
  INTEGER :: printlev = 2
  REAL(KIND = r_std) :: smtot_corr
  REAL(KIND = r_std) :: xios_default_val = 0
  REAL(KIND = r_std) :: zmaxh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: a_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: avan_mod_tab
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: b
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: b_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: check_top_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: d_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: dr_ns, dr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: e
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ep
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: ext_coeff_vegetfrac
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: f
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: fp
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: frac_bare_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: free_drain_coef
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: g1
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: gp
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humcste
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv, humrelv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kfact
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root, kfact_root_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kk, kk_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kk_moy, kk_moy_cpu
  REAL(KIND = r_std), PARAMETER, DIMENSION(nscm_usda) :: ks_usda = (/7128.0_r_std, 3501.6_r_std, 1060.8_r_std, 108.0_r_std, &
&60.0_r_std, 249.6_r_std, 314.4_r_std, 16.8_r_std, 62.4_r_std, 28.8_r_std, 4.8_r_std, 48.0_r_std, 6131.4_r_std/)
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_soiltile
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: max_root_depth
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc_read_current
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl, mcl_cpu
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: natural
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: nslm_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: nvan_mod_tab
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: pcent
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol_ns
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: profil_froz_hydro
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns, profil_froz_hydro_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: qflux_ns, qflux_ns_cpu
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: resolv, resolv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: rhs, rhs_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: rootsink
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ru_ns, ru_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soil_wet_litter, soil_wet_litter_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soil_wet_ns, soil_wet_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist_liquid
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soilmoist_s
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: srhs, srhs_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: stmat, stmat_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: subsinksoil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: tmat, tmat_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc, tmc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_aux
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_dry_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_wet_mea
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter, tmc_litter_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_adry
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_awet
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_field
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_res
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_sat
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_wilt
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tr_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: undermcr, undermcr_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegetmax_soil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegstressv, vegstressv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: water2infilt, water2infilt_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: zdr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: znh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: zwt_force
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: zz
  !$ACC DECLARE COPYIN(ZeroCelsius, allowed_err, alpha_watstress, beta_irrig, deux, do_rsoil, doponds, err_act, huit, icarbon,  &
!$ACC& ifunc, iice, iinterface, imin, inode, iroot, irrig_st, irrigated_soiltile, is_tuzet_hydrol_arch, istruc, kilo_to_unit,  &
!$ACC& ks_usda, lai_irrig_min, lhf, maxaltmax, mille, min_sechiba, min_stomate, min_wind, ncirc, ndepths, nelements,  &
!$ACC& new_watstress, nnobio, nparts, nroot_prof, nscm, nscm_usda, numout, old_irrig_scheme, plev, printlev, quatre, trois, un,  &
!$ACC& undef_sechiba, w_time, xios_default_val, zero)
  !$ACC DECLARE CREATE(a, a_lin, ae_ns, alpha_nudge_mc, avan_mod_tab, b, b_lin, check_cwrr, check_top_ns, d, d_lin, dh, dr_ns,  &
!$ACC& dt_sechiba, dz, e, ep, ext_coeff_vegetfrac, f, fp, fr_center, fr_dT, frac_bare_ns, free_drain_coef, froz_frac_corr, g1,  &
!$ACC& gp, humcste, humrelv, humtot, imax, k, k_lin, kfact, kfact_root, kfact_root_const, kk, kk_moy, mask_soiltile,  &
!$ACC& max_froz_hydro, max_root_depth, mc, mc_read_current, mcl, natural, nslm_root, nvan_mod_tab, ok_freeze_cwrr,  &
!$ACC& ok_hydrol_arch, ok_nudge_mc, ok_thermodynamical_freezing, one_day, pcent, precisol, precisol_ns, pref_soil_veg,  &
!$ACC& profil_froz_hydro, profil_froz_hydro_ns, qflux_ns, resolv, rhs, rootsink, ru_ns, smtot_corr, soil_wet_litter, soil_wet_ns,  &
!$ACC& soilmoist, soilmoist_liquid, soilmoist_s, srhs, stmat, subsinksoil, tmat, tmc, tmc_aux, tmc_litt_dry_mea, tmc_litt_mea,  &
!$ACC& tmc_litt_wet_mea, tmc_litter, tmc_litter_adry, tmc_litter_awet, tmc_litter_field, tmc_litter_res, tmc_litter_sat,  &
!$ACC& tmc_litter_wilt, tr_ns, undermcr, vegetmax_soil, vegstressv, vegtot, vegtot_old, water2infilt, zdr, zmaxh, znh, zwt_force,  &
!$ACC& zz)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil/global.bin', FORM = 'unformatted', STATUS = 'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) alpha_nudge_mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for alpha_nudge_mc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) check_cwrr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for check_cwrr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dt_sechiba
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dt_sechiba. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) fr_center
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fr_center. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) fr_dT
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fr_dT. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) froz_frac_corr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for froz_frac_corr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) imax
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for imax. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kfact_root_const
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kfact_root_const. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) max_froz_hydro
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for max_froz_hydro. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_freeze_cwrr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_freeze_cwrr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_hydrol_arch
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_hydrol_arch. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_nudge_mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_nudge_mc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_thermodynamical_freezing
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_thermodynamical_freezing. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) one_day
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for one_day. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) smtot_corr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for smtot_corr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) zmaxh
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for zmaxh. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(a)) THEN
      ALLOCATE(a(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(a_lin)) THEN
      ALLOCATE(a_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ae_ns)) THEN
      ALLOCATE(ae_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ae_ns_cpu)) THEN
      ALLOCATE(ae_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(avan_mod_tab)) THEN
      ALLOCATE(avan_mod_tab(nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(b)) THEN
      ALLOCATE(b(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(b_lin)) THEN
      ALLOCATE(b_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(check_top_ns)) THEN
      ALLOCATE(check_top_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d)) THEN
      ALLOCATE(d(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d_lin)) THEN
      ALLOCATE(d_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dh)) THEN
      ALLOCATE(dh(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dr_ns)) THEN
      ALLOCATE(dr_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dr_ns_cpu)) THEN
      ALLOCATE(dr_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dz)) THEN
      ALLOCATE(dz(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(e)) THEN
      ALLOCATE(e(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ep)) THEN
      ALLOCATE(ep(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ext_coeff_vegetfrac)) THEN
      ALLOCATE(ext_coeff_vegetfrac(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(f)) THEN
      ALLOCATE(f(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(fp)) THEN
      ALLOCATE(fp(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(frac_bare_ns)) THEN
      ALLOCATE(frac_bare_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(free_drain_coef)) THEN
      ALLOCATE(free_drain_coef(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(g1)) THEN
      ALLOCATE(g1(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(gp)) THEN
      ALLOCATE(gp(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humcste)) THEN
      ALLOCATE(humcste(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humrelv)) THEN
      ALLOCATE(humrelv(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humrelv_cpu)) THEN
      ALLOCATE(humrelv_cpu(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humtot)) THEN
      ALLOCATE(humtot(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k)) THEN
      ALLOCATE(k(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k_lin)) THEN
      ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kfact)) THEN
      ALLOCATE(kfact(nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kfact_root)) THEN
      ALLOCATE(kfact_root(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kfact_root_cpu)) THEN
      ALLOCATE(kfact_root_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kk)) THEN
      ALLOCATE(kk(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kk_cpu)) THEN
      ALLOCATE(kk_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kk_moy)) THEN
      ALLOCATE(kk_moy(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kk_moy_cpu)) THEN
      ALLOCATE(kk_moy_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mask_soiltile)) THEN
      ALLOCATE(mask_soiltile(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(max_root_depth)) THEN
      ALLOCATE(max_root_depth(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc)) THEN
      ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc_cpu)) THEN
      ALLOCATE(mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc_read_current)) THEN
      ALLOCATE(mc_read_current(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mcl)) THEN
      ALLOCATE(mcl(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mcl_cpu)) THEN
      ALLOCATE(mcl_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(natural)) THEN
      ALLOCATE(natural(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(nslm_root)) THEN
      ALLOCATE(nslm_root(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(nvan_mod_tab)) THEN
      ALLOCATE(nvan_mod_tab(nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(pcent)) THEN
      ALLOCATE(pcent(nscm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(precisol)) THEN
      ALLOCATE(precisol(kjpindex, nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(precisol_ns)) THEN
      ALLOCATE(precisol_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(pref_soil_veg)) THEN
      ALLOCATE(pref_soil_veg(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro)) THEN
      ALLOCATE(profil_froz_hydro(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns)) THEN
      ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns_cpu)) THEN
      ALLOCATE(profil_froz_hydro_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(qflux_ns)) THEN
      ALLOCATE(qflux_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(qflux_ns_cpu)) THEN
      ALLOCATE(qflux_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(resolv)) THEN
      ALLOCATE(resolv(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(resolv_cpu)) THEN
      ALLOCATE(resolv_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rhs)) THEN
      ALLOCATE(rhs(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rhs_cpu)) THEN
      ALLOCATE(rhs_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rootsink)) THEN
      ALLOCATE(rootsink(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ru_ns)) THEN
      ALLOCATE(ru_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ru_ns_cpu)) THEN
      ALLOCATE(ru_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_litter)) THEN
      ALLOCATE(soil_wet_litter(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_litter_cpu)) THEN
      ALLOCATE(soil_wet_litter_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_ns)) THEN
      ALLOCATE(soil_wet_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_ns_cpu)) THEN
      ALLOCATE(soil_wet_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist)) THEN
      ALLOCATE(soilmoist(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_liquid)) THEN
      ALLOCATE(soilmoist_liquid(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_s)) THEN
      ALLOCATE(soilmoist_s(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(srhs)) THEN
      ALLOCATE(srhs(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(srhs_cpu)) THEN
      ALLOCATE(srhs_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(stmat)) THEN
      ALLOCATE(stmat(kjpindex, nslm, 3), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(stmat_cpu)) THEN
      ALLOCATE(stmat_cpu(kjpindex, nslm, 3), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(subsinksoil)) THEN
      ALLOCATE(subsinksoil(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmat)) THEN
      ALLOCATE(tmat(kjpindex, nslm, 3), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmat_cpu)) THEN
      ALLOCATE(tmat_cpu(kjpindex, nslm, 3), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc)) THEN
      ALLOCATE(tmc(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_cpu)) THEN
      ALLOCATE(tmc_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_aux)) THEN
      ALLOCATE(tmc_aux(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_dry_mea)) THEN
      ALLOCATE(tmc_litt_dry_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_mea)) THEN
      ALLOCATE(tmc_litt_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_wet_mea)) THEN
      ALLOCATE(tmc_litt_wet_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter)) THEN
      ALLOCATE(tmc_litter(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_cpu)) THEN
      ALLOCATE(tmc_litter_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_adry)) THEN
      ALLOCATE(tmc_litter_adry(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_awet)) THEN
      ALLOCATE(tmc_litter_awet(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_field)) THEN
      ALLOCATE(tmc_litter_field(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_res)) THEN
      ALLOCATE(tmc_litter_res(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_sat)) THEN
      ALLOCATE(tmc_litter_sat(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_wilt)) THEN
      ALLOCATE(tmc_litter_wilt(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tr_ns)) THEN
      ALLOCATE(tr_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(undermcr)) THEN
      ALLOCATE(undermcr(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(undermcr_cpu)) THEN
      ALLOCATE(undermcr_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegetmax_soil)) THEN
      ALLOCATE(vegetmax_soil(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegstressv)) THEN
      ALLOCATE(vegstressv(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegstressv_cpu)) THEN
      ALLOCATE(vegstressv_cpu(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegtot)) THEN
      ALLOCATE(vegtot(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegtot_old)) THEN
      ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(water2infilt)) THEN
      ALLOCATE(water2infilt(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(water2infilt_cpu)) THEN
      ALLOCATE(water2infilt_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(zdr)) THEN
      ALLOCATE(zdr(0 : nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(znh)) THEN
      ALLOCATE(znh(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(zwt_force)) THEN
      ALLOCATE(zwt_force(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(zz)) THEN
      ALLOCATE(zz(nslm), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) a
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for a. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) a_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for a_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ae_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ae_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) avan_mod_tab
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for avan_mod_tab. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) b
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for b. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) b_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for b_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) check_top_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for check_top_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) d
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for d. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) d_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for d_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dh
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dh. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dr_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dr_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) e
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for e. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ep
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ep. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ext_coeff_vegetfrac
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ext_coeff_vegetfrac. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) f
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for f. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) fp
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fp. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) frac_bare_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for frac_bare_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) free_drain_coef
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for free_drain_coef. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) g1
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for g1. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) gp
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for gp. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humcste
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humcste. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humrelv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humrelv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humtot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humtot. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) k
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for k. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) k_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for k_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kfact
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kfact. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kfact_root
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kfact_root. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kk
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kk. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kk_moy
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kk_moy. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mask_soiltile
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mask_soiltile. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) max_root_depth
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for max_root_depth. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mc_read_current
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mc_read_current. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcl
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcl. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) natural
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for natural. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) nslm_root
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for nslm_root. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) nvan_mod_tab
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for nvan_mod_tab. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) pcent
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for pcent. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precisol
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precisol. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precisol_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precisol_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) pref_soil_veg
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for pref_soil_veg. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) profil_froz_hydro
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for profil_froz_hydro. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) profil_froz_hydro_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for profil_froz_hydro_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) qflux_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for qflux_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) resolv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for resolv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) rhs
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for rhs. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) rootsink
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for rootsink. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ru_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ru_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soil_wet_litter
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soil_wet_litter. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soil_wet_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soil_wet_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist_liquid
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist_liquid. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist_s
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist_s. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) srhs
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for srhs. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) stmat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for stmat. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) subsinksoil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for subsinksoil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmat. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_aux
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_aux. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_dry_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_dry_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_wet_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_wet_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_adry
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_adry. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_awet
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_awet. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_field
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_field. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_res
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_res. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_sat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_sat. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_wilt
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_wilt. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tr_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tr_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) undermcr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for undermcr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegetmax_soil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegetmax_soil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegstressv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegstressv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegtot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegtot. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegtot_old
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegtot_old. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) water2infilt
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for water2infilt. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) zdr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for zdr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) znh
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for znh. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) zwt_force
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for zwt_force. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) zz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for zz. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(a, a_lin, ae_ns, alpha_nudge_mc, avan_mod_tab, b, b_lin, check_cwrr, check_top_ns, d, d_lin, dh, dr_ns,  &
!$ACC& dt_sechiba, dz, e, ep, ext_coeff_vegetfrac, f, fp, fr_center, fr_dT, frac_bare_ns, free_drain_coef, froz_frac_corr, g1,  &
!$ACC& gp, humcste, humrelv, humtot, imax, k, k_lin, kfact, kfact_root, kfact_root_const, kk, kk_moy, mask_soiltile,  &
!$ACC& max_froz_hydro, max_root_depth, mc, mc_read_current, mcl, natural, nslm_root, nvan_mod_tab, ok_freeze_cwrr,  &
!$ACC& ok_hydrol_arch, ok_nudge_mc, ok_thermodynamical_freezing, one_day, pcent, precisol, precisol_ns, pref_soil_veg,  &
!$ACC& profil_froz_hydro, profil_froz_hydro_ns, qflux_ns, resolv, rhs, rootsink, ru_ns, smtot_corr, soil_wet_litter, soil_wet_ns,  &
!$ACC& soilmoist, soilmoist_liquid, soilmoist_s, srhs, stmat, subsinksoil, tmat, tmc, tmc_aux, tmc_litt_dry_mea, tmc_litt_mea,  &
!$ACC& tmc_litt_wet_mea, tmc_litter, tmc_litter_adry, tmc_litter_awet, tmc_litter_field, tmc_litter_res, tmc_litter_sat,  &
!$ACC& tmc_litter_wilt, tr_ns, undermcr, vegetmax_soil, vegstressv, vegtot, vegtot_old, water2infilt, zdr, zmaxh, znh, zwt_force,  &
!$ACC& zz)
  END SUBROUTINE declaration_initialization
END MODULE module_global
