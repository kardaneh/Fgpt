
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
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std) :: alpha_nudge_mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc_read_current
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_aux, tmc_aux_cpu
  !$ACC DECLARE COPYIN(trois, huit)
  !$ACC DECLARE CREATE(dz, mc_read_current, tmc_aux, alpha_nudge_mc)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    CALL random_number(alpha_nudge_mc)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(mc_read_current(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(tmc_aux(kjpindex, nstm), tmc_aux_cpu(kjpindex, nstm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(dz)
    CALL random_number(mc_read_current)
    CALL random_number(tmc_aux)
    !$ACC UPDATE DEVICE(dz, mc_read_current, tmc_aux, alpha_nudge_mc)
  END SUBROUTINE initialization
END MODULE module_global
