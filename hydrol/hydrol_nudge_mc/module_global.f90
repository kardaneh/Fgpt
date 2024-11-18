
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
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std) :: alpha_nudge_mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc_read_current
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_aux, tmc_aux_cpu
  !$ACC DECLARE COPYIN(huit, trois)
  !$ACC DECLARE CREATE(alpha_nudge_mc, dz, mc_read_current, tmc_aux)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_nudge_mc/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) alpha_nudge_mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for alpha_nudge_mc. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(dz)) THEN
      ALLOCATE(dz(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc_read_current)) THEN
      ALLOCATE(mc_read_current(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_aux)) THEN
      ALLOCATE(tmc_aux(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_aux_cpu)) THEN
      ALLOCATE(tmc_aux_cpu(kjpindex, nstm), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) dz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mc_read_current
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mc_read_current. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_aux
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_aux. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(alpha_nudge_mc, dz, mc_read_current, tmc_aux)
  END SUBROUTINE declaration_initialization
END MODULE module_global
