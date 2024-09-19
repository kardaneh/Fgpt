
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
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: ZeroCelsius = 273.15
  LOGICAL :: ok_freeze_cwrr
  LOGICAL :: check_cwrr
  INTEGER(KIND = i_std) :: numout = 6
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std) :: one_day
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: kfact
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  !$ACC DECLARE COPYIN(trois, un, huit, min_sechiba, zero, deux, ZeroCelsius, numout)
  !$ACC DECLARE CREATE(dz, ok_freeze_cwrr, kfact, k, kfact_root, check_cwrr, mc, dt_sechiba, one_day)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    ok_freeze_cwrr = .TRUE.
    check_cwrr = .TRUE.
    CALL random_number(dt_sechiba)
    CALL random_number(one_day)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(kfact(nslm, kjpindex), STAT = ier)
    ALLOCATE(k(kjpindex, nslm), STAT = ier)
    ALLOCATE(kfact_root(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(dz)
    CALL random_number(kfact)
    CALL random_number(k)
    CALL random_number(kfact_root)
    CALL random_number(mc)
    !$ACC UPDATE DEVICE(dz, ok_freeze_cwrr, kfact, k, kfact_root, check_cwrr, mc, dt_sechiba, one_day)
  END SUBROUTINE initialization
END MODULE module_global
