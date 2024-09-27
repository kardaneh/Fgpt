
MODULE module_global
  IMPLICIT NONE
  INTEGER, PARAMETER :: i_std = 4
  INTEGER, PARAMETER :: r_std = 8
  INTEGER(KIND = i_std), PARAMETER :: nslm = 11
  INTEGER(KIND = i_std), PARAMETER :: nvm = 13
  INTEGER(KIND = i_std), PARAMETER :: nstm = 3
  INTEGER(KIND = i_std), PARAMETER :: kjpindex = 4716
  INTEGER :: ier
  INTEGER :: seed(64) = 1
  INTEGER(KIND = i_std), PARAMETER :: iice = 1
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  INTEGER(KIND = i_std), PARAMETER :: nnobio = 1
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  INTEGER :: imax
  LOGICAL :: ok_freeze_cwrr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc, tmc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist, soilmoist_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_mea, tmc_litt_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soilmoist_s, soilmoist_s_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_awet
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: profil_froz_hydro, profil_froz_hydro_cpu
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_soiltile
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist_liquid, soilmoist_liquid_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_wet_mea, tmc_litt_wet_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ru_ns, ru_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_adry
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soil_wet_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_res
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_dry_mea, tmc_litt_dry_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot, humtot_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: dr_ns, dr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_sat
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegstressv
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: frac_bare_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv, humrelv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soil_wet_litter
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: subsinksoil
  !$ACC DECLARE COPYIN(iice, un, min_sechiba, zero, trois, huit, nnobio, imin)
  !$ACC DECLARE CREATE(ae_ns, tmc, profil_froz_hydro_ns, imax, soilmoist, tmc_litt_mea, soilmoist_s, vegtot, k_lin, mc, tmc_litter_awet, dz, profil_froz_hydro, mask_soiltile, mcl, dh, soilmoist_liquid, tmc_litt_wet_mea, ru_ns, vegtot_old, tmc_litter_adry, soil_wet_ns, tmc_litter, ok_freeze_cwrr, tmc_litter_res, tmc_litt_dry_mea, humtot, dr_ns, tmc_litter_sat, vegstressv, frac_bare_ns, humrelv, soil_wet_litter, subsinksoil)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    imax = 2
    ok_freeze_cwrr = .TRUE.
    ALLOCATE(ae_ns(kjpindex, nstm), ae_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc(kjpindex, nstm), tmc_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(soilmoist(kjpindex, nslm), soilmoist_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(tmc_litt_mea(kjpindex), tmc_litt_mea_cpu(kjpindex), STAT = ier)
    ALLOCATE(soilmoist_s(kjpindex, nslm, nstm), soilmoist_s_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(vegtot(kjpindex), STAT = ier)
    ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmc_litter_awet(kjpindex, nstm), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(profil_froz_hydro(kjpindex, nslm), profil_froz_hydro_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(mask_soiltile(kjpindex, nstm), STAT = ier)
    ALLOCATE(mcl(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(dh(nslm), STAT = ier)
    ALLOCATE(soilmoist_liquid(kjpindex, nslm), soilmoist_liquid_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(tmc_litt_wet_mea(kjpindex), tmc_litt_wet_mea_cpu(kjpindex), STAT = ier)
    ALLOCATE(ru_ns(kjpindex, nstm), ru_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    ALLOCATE(tmc_litter_adry(kjpindex, nstm), STAT = ier)
    ALLOCATE(soil_wet_ns(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmc_litter(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litter_res(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litt_dry_mea(kjpindex), tmc_litt_dry_mea_cpu(kjpindex), STAT = ier)
    ALLOCATE(humtot(kjpindex), humtot_cpu(kjpindex), STAT = ier)
    ALLOCATE(dr_ns(kjpindex, nstm), dr_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc_litter_sat(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegstressv(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(frac_bare_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(humrelv(kjpindex, nvm, nstm), humrelv_cpu(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(soil_wet_litter(kjpindex, nstm), STAT = ier)
    ALLOCATE(subsinksoil(kjpindex), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(ae_ns)
    CALL random_number(tmc)
    CALL random_number(profil_froz_hydro_ns)
    CALL random_number(soilmoist)
    CALL random_number(tmc_litt_mea)
    CALL random_number(soilmoist_s)
    CALL random_number(vegtot)
    CALL random_number(k_lin)
    CALL random_number(mc)
    CALL random_number(tmc_litter_awet)
    CALL random_number(dz)
    CALL random_number(profil_froz_hydro)
    mask_soiltile = 2
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
    !$ACC UPDATE DEVICE(ae_ns, tmc, profil_froz_hydro_ns, imax, soilmoist, tmc_litt_mea, soilmoist_s, vegtot, k_lin, mc, tmc_litter_awet, dz, profil_froz_hydro, mask_soiltile, mcl, dh, soilmoist_liquid, tmc_litt_wet_mea, ru_ns, vegtot_old, tmc_litter_adry, soil_wet_ns, tmc_litter, ok_freeze_cwrr, tmc_litter_res, tmc_litt_dry_mea, humtot, dr_ns, tmc_litter_sat, vegstressv, frac_bare_ns, humrelv, soil_wet_litter, subsinksoil)
  END SUBROUTINE initialization
END MODULE module_global
