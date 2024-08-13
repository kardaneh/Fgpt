
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
  INTEGER :: printlev = 2
  INTEGER(KIND = i_std) :: numout = 6
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot, humtot_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: water2infilt, water2infilt_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc, tmc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: resdist, resdist_cpu
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  !$ACC DECLARE COPYIN(trois, printlev, huit, zero, numout, min_sechiba)
  !$ACC DECLARE CREATE(humtot, dz, vegtot_old, mc, water2infilt, tmc, vegtot, resdist, pref_soil_veg)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    ALLOCATE(humtot(kjpindex), humtot_cpu(kjpindex), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(water2infilt(kjpindex, nstm), water2infilt_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(tmc(kjpindex, nstm), tmc_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegtot(kjpindex), STAT = ier)
    ALLOCATE(resdist(kjpindex, nstm), resdist_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(pref_soil_veg(nvm), STAT = ier)
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
    !$ACC UPDATE DEVICE(humtot, dz, vegtot_old, mc, water2infilt, tmc, vegtot, resdist, pref_soil_veg)
  END SUBROUTINE initialization
END MODULE module_global
