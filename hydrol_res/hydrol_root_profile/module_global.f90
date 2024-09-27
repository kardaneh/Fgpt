
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
  INTEGER(KIND = i_std), PARAMETER :: ndepths = 2
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  INTEGER(KIND = i_std), PARAMETER :: istruc = 1
  INTEGER(KIND = i_std), PARAMETER :: nroot_prof = 2
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  INTEGER(KIND = i_std), PARAMETER :: inode = 1
  INTEGER(KIND = i_std), PARAMETER :: iinterface = 2
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  INTEGER(KIND = i_std), PARAMETER :: ifunc = 2
  INTEGER(KIND = i_std) :: numout = 6
  INTEGER(KIND = i_std) :: plev = 0
  INTEGER(KIND = i_std) :: err_act = 1
  REAL(KIND = r_std) :: maxaltmax = 2.
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: zdr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: znh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: max_root_depth
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humcste
  !$ACC DECLARE COPYIN(numout, plev, ndepths, zero, istruc, nroot_prof, un, inode, err_act, iinterface, min_sechiba, ifunc, maxaltmax)
  !$ACC DECLARE CREATE(zdr, znh, max_root_depth, humcste)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    ALLOCATE(zdr(0 : nslm), STAT = ier)
    ALLOCATE(znh(nslm), STAT = ier)
    ALLOCATE(max_root_depth(nvm), STAT = ier)
    ALLOCATE(humcste(nvm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(zdr)
    CALL random_number(znh)
    CALL random_number(max_root_depth)
    CALL random_number(humcste)
    !$ACC UPDATE DEVICE(zdr, znh, max_root_depth, humcste)
  END SUBROUTINE initialization
END MODULE module_global
