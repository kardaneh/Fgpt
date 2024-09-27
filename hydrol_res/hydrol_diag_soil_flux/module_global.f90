
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
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  INTEGER(KIND = i_std) :: numout = 6
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: dr_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: qflux_ns, qflux_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: rootsink
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: check_top_ns, check_top_ns_cpu
  !$ACC DECLARE COPYIN(numout, min_sechiba, trois, huit)
  !$ACC DECLARE CREATE(dz, dr_ns, qflux_ns, mcl, rootsink, dt_sechiba, check_top_ns)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    CALL random_number(dt_sechiba)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(dr_ns(kjpindex, nstm), STAT = ier)
    ALLOCATE(qflux_ns(kjpindex, nslm, nstm), qflux_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(mcl(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(rootsink(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(check_top_ns(kjpindex, nstm), check_top_ns_cpu(kjpindex, nstm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(dz)
    CALL random_number(dr_ns)
    CALL random_number(qflux_ns)
    CALL random_number(mcl)
    CALL random_number(rootsink)
    CALL random_number(check_top_ns)
    !$ACC UPDATE DEVICE(dz, dr_ns, qflux_ns, mcl, rootsink, dt_sechiba, check_top_ns)
  END SUBROUTINE initialization
END MODULE module_global
