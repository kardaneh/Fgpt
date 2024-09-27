
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
  REAL(KIND = r_std), PARAMETER :: ZeroCelsius = 273.15
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  INTEGER(KIND = i_std) :: numout = 6
  REAL(KIND = r_std) :: one_day
  LOGICAL :: ok_freeze_cwrr
  LOGICAL :: check_cwrr
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kfact
  !$ACC DECLARE COPYIN(numout, ZeroCelsius, zero, min_sechiba, deux, un, trois, huit)
  !$ACC DECLARE CREATE(mc, one_day, dz, kfact_root, ok_freeze_cwrr, k, kfact, check_cwrr, dt_sechiba)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    CALL random_number(one_day)
    ok_freeze_cwrr = .TRUE.
    check_cwrr = .TRUE.
    CALL random_number(dt_sechiba)
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(kfact_root(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(k(kjpindex, nslm), STAT = ier)
    ALLOCATE(kfact(nslm, kjpindex), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(mc)
    CALL random_number(dz)
    CALL random_number(kfact_root)
    CALL random_number(k)
    CALL random_number(kfact)
    !$ACC UPDATE DEVICE(mc, one_day, dz, kfact_root, ok_freeze_cwrr, k, kfact, check_cwrr, dt_sechiba)
  END SUBROUTINE initialization
END MODULE module_global
