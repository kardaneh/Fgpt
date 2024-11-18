
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
  REAL(KIND = r_std), PARAMETER :: deux = 2._r_std
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: w_time = 1.0_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std) :: one_day
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: a
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: d
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: e, e_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ep, ep_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: f, f_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: fp, fp_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: free_drain_coef
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: g1, g1_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: gp, gp_cpu
  !$ACC DECLARE COPYIN(deux, huit, trois, un, w_time, zero)
  !$ACC DECLARE CREATE(a, d, dt_sechiba, dz, e, ep, f, fp, free_drain_coef, g1, gp, one_day)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_setup/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) dt_sechiba
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dt_sechiba. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) one_day
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for one_day. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(a)) THEN
      ALLOCATE(a(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(d)) THEN
      ALLOCATE(d(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dz)) THEN
      ALLOCATE(dz(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(e)) THEN
      ALLOCATE(e(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(e_cpu)) THEN
      ALLOCATE(e_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ep)) THEN
      ALLOCATE(ep(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ep_cpu)) THEN
      ALLOCATE(ep_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(f)) THEN
      ALLOCATE(f(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(f_cpu)) THEN
      ALLOCATE(f_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(fp)) THEN
      ALLOCATE(fp(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(fp_cpu)) THEN
      ALLOCATE(fp_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(free_drain_coef)) THEN
      ALLOCATE(free_drain_coef(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(g1)) THEN
      ALLOCATE(g1(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(g1_cpu)) THEN
      ALLOCATE(g1_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(gp)) THEN
      ALLOCATE(gp(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(gp_cpu)) THEN
      ALLOCATE(gp_cpu(kjpindex, nslm), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) a
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for a. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) d
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for d. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) e
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for e. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ep
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ep. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) f
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for f. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) fp
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fp. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) free_drain_coef
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for free_drain_coef. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) g1
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for g1. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) gp
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for gp. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(a, d, dt_sechiba, dz, e, ep, f, fp, free_drain_coef, g1, gp, one_day)
  END SUBROUTINE declaration_initialization
END MODULE module_global
