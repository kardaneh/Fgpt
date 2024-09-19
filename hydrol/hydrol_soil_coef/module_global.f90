
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
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  INTEGER :: imax
  LOGICAL :: ok_freeze_cwrr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d, d_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: a_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a, a_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k, k_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: d_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: b, b_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: b_lin
  !$ACC DECLARE COPYIN(imin, zero)
  !$ACC DECLARE CREATE(profil_froz_hydro_ns, d, imax, ok_freeze_cwrr, a_lin, a, k, kfact_root, mc, k_lin, d_lin, b, b_lin)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    imax = 2
    ok_freeze_cwrr = .TRUE.
    ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(d(kjpindex, nslm), d_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(a_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(a(kjpindex, nslm), a_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(k(kjpindex, nslm), k_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(kfact_root(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(d_lin(imin : imax, nslm, kjpindex), STAT = ier)
    ALLOCATE(b(kjpindex, nslm), b_cpu(kjpindex, nslm), STAT = ier)
    ALLOCATE(b_lin(imin : imax, nslm, kjpindex), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(profil_froz_hydro_ns)
    CALL random_number(d)
    CALL random_number(a_lin)
    CALL random_number(a)
    CALL random_number(k)
    CALL random_number(kfact_root)
    CALL random_number(mc)
    CALL random_number(k_lin)
    CALL random_number(d_lin)
    CALL random_number(b)
    CALL random_number(b_lin)
    !$ACC UPDATE DEVICE(profil_froz_hydro_ns, d, imax, ok_freeze_cwrr, a_lin, a, k, kfact_root, mc, k_lin, d_lin, b, b_lin)
  END SUBROUTINE initialization
END MODULE module_global
