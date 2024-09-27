
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
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: resolv
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl, mcl_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: tmat
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: rhs
  !$ACC DECLARE CREATE(resolv, mcl, tmat, rhs)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    ALLOCATE(resolv(kjpindex), STAT = ier)
    ALLOCATE(mcl(kjpindex, nslm, nstm), mcl_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmat(kjpindex, nslm, 3), STAT = ier)
    ALLOCATE(rhs(kjpindex, nslm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    resolv = .TRUE.
    CALL random_number(mcl)
    CALL random_number(tmat)
    CALL random_number(rhs)
    !$ACC UPDATE DEVICE(resolv, mcl, tmat, rhs)
  END SUBROUTINE initialization
END MODULE module_global
