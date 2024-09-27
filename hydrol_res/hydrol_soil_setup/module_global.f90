
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
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: w_time = 1.0_r_std
  REAL(KIND = r_std) :: one_day
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: fp, fp_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: free_drain_coef
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: g1, g1_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: f, f_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ep, ep_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: e, e_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: gp, gp_cpu
  !$ACC DECLARE COPYIN(un, zero, trois, huit, deux, w_time)
  !$ACC DECLARE CREATE(one_day, fp, dt_sechiba, free_drain_coef, dz, d, g1, f, ep, a, e, gp)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    CALL random_number(one_day)
    CALL random_number(dt_sechiba)
    ALLOCATE(fp(kjpindex, nslm), fp_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(free_drain_coef(kjpindex, nstm), STAT = ier)
    ALLOCATE(dz(nslm), STAT = ier)
    ALLOCATE(d(kjpindex, nslm), STAT = ier)
    ALLOCATE(g1(kjpindex, nslm), g1_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(f(kjpindex, nslm), f_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(ep(kjpindex, nslm), ep_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(a(kjpindex, nslm), STAT = ier)
    ALLOCATE(e(kjpindex, nslm), e_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(gp(kjpindex, nslm), gp_cpu(kjpindex, nslm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(fp)
    CALL random_number(free_drain_coef)
    CALL random_number(dz)
    CALL random_number(d)
    CALL random_number(g1)
    CALL random_number(f)
    CALL random_number(ep)
    CALL random_number(a)
    CALL random_number(e)
    CALL random_number(gp)
    !$ACC UPDATE DEVICE(one_day, fp, dt_sechiba, free_drain_coef, dz, d, g1, f, ep, a, e, gp)
  END SUBROUTINE initialization
END MODULE module_global
