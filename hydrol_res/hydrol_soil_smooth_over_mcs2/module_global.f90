
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
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  LOGICAL :: check_cwrr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  !$ACC DECLARE COPYIN(zero, trois, huit)
  !$ACC DECLARE CREATE(mc, dz, check_cwrr)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    check_cwrr = .TRUE.
    ALLOCATE(mc(kjpindex, nslm, nstm), mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(mc)
    CALL random_number(dz)
    !$ACC UPDATE DEVICE(mc, dz, check_cwrr)
  END SUBROUTINE initialization
END MODULE module_global
