
MODULE module_global
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
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  INTEGER :: printlev = 2
  INTEGER(KIND = i_std) :: numout = 6
  LOGICAL :: ok_bare_soil_new
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: water2infilt
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: resdist
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_veget, mask_veget_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: frac_bare_ns, frac_bare_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegetmax_soil, vegetmax_soil_cpu
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_soiltile, mask_soiltile_cpu
  !$ACC DECLARE COPYIN(trois, printlev, huit, zero, numout, min_sechiba, un)
  !$ACC DECLARE CREATE(humtot, dz, vegtot_old, mc, water2infilt, tmc, vegtot, resdist, pref_soil_veg, ok_bare_soil_new, mask_veget, frac_bare_ns, vegetmax_soil, mask_soiltile)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    ok_bare_soil_new = .TRUE.
    ALLOCATE(humtot(kjpindex), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(water2infilt(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegtot(kjpindex), STAT = ier)
    ALLOCATE(resdist(kjpindex, nstm), STAT = ier)
    ALLOCATE(pref_soil_veg(nvm), STAT = ier)
    ALLOCATE(mask_veget(kjpindex, nvm), mask_veget_cpu(kjpindex, nvm), STAT = ier)
    ALLOCATE(frac_bare_ns(kjpindex, nstm), frac_bare_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegetmax_soil(kjpindex, nvm, nstm), vegetmax_soil_cpu(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(mask_soiltile(kjpindex, nstm), mask_soiltile_cpu(kjpindex, nstm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(humtot)
    CALL random_number(dz)
    CALL random_number(vegtot_old)
    CALL random_number(mc)
    CALL random_number(water2infilt)
    CALL random_number(tmc)
    CALL random_number(vegtot)
    CALL random_number(resdist)
    pref_soil_veg = 2
    mask_veget = 2
    CALL random_number(frac_bare_ns)
    CALL random_number(vegetmax_soil)
    mask_soiltile = 2
    !$ACC UPDATE DEVICE(humtot, dz, vegtot_old, mc, water2infilt, tmc, vegtot, resdist, pref_soil_veg, ok_bare_soil_new, mask_veget, frac_bare_ns, vegetmax_soil, mask_soiltile)
  END SUBROUTINE initialization
END MODULE module_global
