
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
  REAL(KIND = r_std), PARAMETER :: ZeroCelsius = 273.15
  REAL(KIND = r_std), PARAMETER :: lhf = 0.3336 * 1.E6
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std) :: fr_center
  REAL(KIND = r_std) :: fr_dT
  REAL(KIND = r_std) :: froz_frac_corr
  REAL(KIND = r_std) :: max_froz_hydro
  LOGICAL :: ok_thermodynamical_freezing
  REAL(KIND = r_std) :: smtot_corr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns, profil_froz_hydro_ns_cpu
  !$ACC DECLARE COPYIN(ZeroCelsius, lhf, min_sechiba, zero)
  !$ACC DECLARE CREATE(dh, fr_center, fr_dT, froz_frac_corr, max_froz_hydro, mc, ok_thermodynamical_freezing,  &
!$ACC& profil_froz_hydro_ns, smtot_corr)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_froz/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) fr_center
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fr_center. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) fr_dT
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for fr_dT. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) froz_frac_corr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for froz_frac_corr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) max_froz_hydro
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for max_froz_hydro. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_thermodynamical_freezing
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_thermodynamical_freezing. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) smtot_corr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for smtot_corr. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(dh)) THEN
      ALLOCATE(dh(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc)) THEN
      ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns)) THEN
      ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns_cpu)) THEN
      ALLOCATE(profil_froz_hydro_ns_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) dh
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dh. ', ' IOSTAT : ', ier
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
    !$ACC UPDATE DEVICE(dh, fr_center, fr_dT, froz_frac_corr, max_froz_hydro, mc, ok_thermodynamical_freezing,  &
!$ACC& profil_froz_hydro_ns, smtot_corr)
  END SUBROUTINE declaration_initialization
END MODULE module_global
