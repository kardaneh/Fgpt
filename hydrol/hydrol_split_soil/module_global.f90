
MODULE module_global
  USE ioipsl_para, ONLY: ipslerr_p
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
  REAL(KIND = r_std), PARAMETER :: allowed_err = 2.0E-8_r_std
  REAL(KIND = r_std), PARAMETER :: kilo_to_unit = 1.0E03
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  LOGICAL :: check_cwrr
  REAL(KIND = r_std) :: dt_sechiba
  LOGICAL :: is_tuzet_hydrol_arch = .FALSE.
  INTEGER(KIND = i_std) :: numout = 6
  LOGICAL :: ok_hydrol_arch
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol_ns, precisol_ns_cpu
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: rootsink, rootsink_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tr_ns, tr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegetmax_soil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  !$ACC DECLARE COPYIN(allowed_err, is_tuzet_hydrol_arch, kilo_to_unit, min_sechiba, numout, zero)
  !$ACC DECLARE CREATE(ae_ns, check_cwrr, dt_sechiba, humrelv, ok_hydrol_arch, precisol, precisol_ns, pref_soil_veg, rootsink,  &
!$ACC& tr_ns, vegetmax_soil, vegtot)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_split_soil/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) check_cwrr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for check_cwrr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dt_sechiba
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dt_sechiba. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_hydrol_arch
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_hydrol_arch. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(ae_ns)) THEN
      ALLOCATE(ae_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ae_ns_cpu)) THEN
      ALLOCATE(ae_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humrelv)) THEN
      ALLOCATE(humrelv(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(precisol)) THEN
      ALLOCATE(precisol(kjpindex, nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(precisol_ns)) THEN
      ALLOCATE(precisol_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(precisol_ns_cpu)) THEN
      ALLOCATE(precisol_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(pref_soil_veg)) THEN
      ALLOCATE(pref_soil_veg(nvm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rootsink)) THEN
      ALLOCATE(rootsink(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(rootsink_cpu)) THEN
      ALLOCATE(rootsink_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tr_ns)) THEN
      ALLOCATE(tr_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tr_ns_cpu)) THEN
      ALLOCATE(tr_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegetmax_soil)) THEN
      ALLOCATE(vegetmax_soil(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegtot)) THEN
      ALLOCATE(vegtot(kjpindex), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) ae_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ae_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humrelv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humrelv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precisol
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precisol. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precisol_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precisol_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) pref_soil_veg
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for pref_soil_veg. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) rootsink
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for rootsink. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tr_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tr_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegetmax_soil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegetmax_soil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegtot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegtot. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(ae_ns, check_cwrr, dt_sechiba, humrelv, ok_hydrol_arch, precisol, precisol_ns, pref_soil_veg, rootsink,  &
!$ACC& tr_ns, vegetmax_soil, vegtot)
  END SUBROUTINE declaration_initialization
END MODULE module_global
