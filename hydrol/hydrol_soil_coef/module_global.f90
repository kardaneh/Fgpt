
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
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  INTEGER :: imax
  LOGICAL :: ok_freeze_cwrr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a, a_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: a_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: b, b_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: b_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d, d_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: d_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: k, k_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: kfact_root
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns
  !$ACC DECLARE COPYIN(imin, zero)
  !$ACC DECLARE CREATE(a, a_lin, b, b_lin, d, d_lin, imax, k, k_lin, kfact_root, mc, ok_freeze_cwrr, profil_froz_hydro_ns)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_coef/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) imax
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for imax. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_freeze_cwrr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_freeze_cwrr. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(a)) THEN
      ALLOCATE(a(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(a_cpu)) THEN
      ALLOCATE(a_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(a_lin)) THEN
      ALLOCATE(a_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(b)) THEN
      ALLOCATE(b(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(b_cpu)) THEN
      ALLOCATE(b_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(b_lin)) THEN
      ALLOCATE(b_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d)) THEN
      ALLOCATE(d(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d_cpu)) THEN
      ALLOCATE(d_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d_lin)) THEN
      ALLOCATE(d_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k)) THEN
      ALLOCATE(k(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k_cpu)) THEN
      ALLOCATE(k_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k_lin)) THEN
      ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(kfact_root)) THEN
      ALLOCATE(kfact_root(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc)) THEN
      ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns)) THEN
      ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) a
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for a. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) a_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for a_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) b
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for b. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) b_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for b_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) d
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for d. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) d_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for d_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) k
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for k. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) k_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for k_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) kfact_root
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for kfact_root. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) profil_froz_hydro_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for profil_froz_hydro_ns. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(a, a_lin, b, b_lin, d, d_lin, imax, k, k_lin, kfact_root, mc, ok_freeze_cwrr, profil_froz_hydro_ns)
  END SUBROUTINE declaration_initialization
END MODULE module_global
