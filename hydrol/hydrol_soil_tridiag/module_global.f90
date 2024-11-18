
MODULE module_global
  IMPLICIT NONE
  INTEGER, PARAMETER :: i_std = 4
  INTEGER, PARAMETER :: r_std = 8
  INTEGER(KIND = i_std), PARAMETER :: nsnow = 3
  INTEGER(KIND = i_std), PARAMETER :: nslm = 11
  INTEGER(KIND = i_std), PARAMETER :: nvm = 15
  INTEGER(KIND = i_std), PARAMETER :: nstm = 3
  INTEGER(KIND = i_std), PARAMETER :: kjpindex = 4717
  INTEGER :: ier
  INTEGER(KIND = i_std) :: ic0, ic
  REAL(KIND = r_std) :: icr, start_time, stop_time
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl, mcl_cpu
  LOGICAL, ALLOCATABLE, DIMENSION(:) :: resolv
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: rhs
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: tmat
  !$ACC DECLARE CREATE(mcl, resolv, rhs, tmat)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_tridiag/global.bin', FORM = 'unformatted', STATUS &
&= 'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    IF (.NOT. ALLOCATED(mcl)) THEN
      ALLOCATE(mcl(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mcl_cpu)) THEN
      ALLOCATE(mcl_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(resolv)) THEN
      ALLOCATE(resolv(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rhs)) THEN
      ALLOCATE(rhs(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmat)) THEN
      ALLOCATE(tmat(kjpindex, nslm, 3), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) mcl
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcl. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) resolv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for resolv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) rhs
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for rhs. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmat. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(mcl, resolv, rhs, tmat)
  END SUBROUTINE declaration_initialization
END MODULE module_global
