
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
  INTEGER(KIND = i_std), PARAMETER :: iice = 1
  INTEGER(KIND = i_std), PARAMETER :: nnobio = 1
  REAL(KIND = r_std), PARAMETER :: tp_00 = 273.15
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std) :: dt_sechiba
  REAL(KIND = r_std) :: max_snow_age = 50._r_std
  REAL(KIND = r_std) :: omg1 = 7.0
  REAL(KIND = r_std) :: omg2 = 4.0
  REAL(KIND = r_std) :: one_day
  REAL(KIND = r_std) :: snow_trans = 0.2_r_std
  REAL(KIND = r_std) :: snow_trans_nobio = 1.0_r_std
  !$ACC DECLARE COPYIN(iice, max_snow_age, nnobio, omg1, omg2, snow_trans, snow_trans_nobio, tp_00, un, zero)
  !$ACC DECLARE CREATE(dt_sechiba, one_day)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/explicitsnow_age/global.bin', FORM = 'unformatted', STATUS = &
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
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(dt_sechiba, one_day)
  END SUBROUTINE declaration_initialization
END MODULE module_global
