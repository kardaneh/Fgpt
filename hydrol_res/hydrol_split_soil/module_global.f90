
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
  REAL(KIND = r_std), PARAMETER :: allowed_err = 2.0E-8_r_std
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  REAL(KIND = r_std), PARAMETER :: kilo_to_unit = 1.0E03
  INTEGER(KIND = i_std) :: numout = 6
  REAL(KIND = r_std) :: dt_sechiba
  LOGICAL :: is_tuzet_hydrol_arch = .FALSE.
  LOGICAL :: check_cwrr
  LOGICAL :: ok_hydrol_arch
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tr_ns, tr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: precisol_ns, precisol_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: rootsink, rootsink_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegetmax_soil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:) :: pref_soil_veg
  !$ACC DECLARE COPYIN(numout, allowed_err, min_sechiba, is_tuzet_hydrol_arch, zero, kilo_to_unit)
  !$ACC DECLARE CREATE(ae_ns, precisol, tr_ns, vegtot, dt_sechiba, precisol_ns, rootsink, check_cwrr, ok_hydrol_arch, vegetmax_soil, humrelv, pref_soil_veg)
  CONTAINS
  SUBROUTINE declarations
    CALL random_seed(put = seed)
    WRITE(*, *) '--- add the declarations in module global ---'
    CALL random_number(dt_sechiba)
    check_cwrr = .TRUE.
    ok_hydrol_arch = .TRUE.
    ALLOCATE(ae_ns(kjpindex, nstm), ae_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(precisol(kjpindex, nvm), STAT = ier)
    ALLOCATE(tr_ns(kjpindex, nstm), tr_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(vegtot(kjpindex), STAT = ier)
    ALLOCATE(precisol_ns(kjpindex, nstm), precisol_ns_cpu(kjpindex, nstm), STAT = ier)
    ALLOCATE(rootsink(kjpindex, nslm, nstm), rootsink_cpu(kjpindex, nslm, nstm), STAT = ier)
    ALLOCATE(vegetmax_soil(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(humrelv(kjpindex, nvm, nstm), STAT = ier)
    ALLOCATE(pref_soil_veg(nvm), STAT = ier)
  END SUBROUTINE declarations
  SUBROUTINE initialization
    CALL random_seed(put = seed)
    WRITE(*, *) '--- initialization of global variables in module global ---'
    CALL random_number(ae_ns)
    CALL random_number(precisol)
    CALL random_number(tr_ns)
    CALL random_number(vegtot)
    CALL random_number(precisol_ns)
    CALL random_number(rootsink)
    CALL random_number(vegetmax_soil)
    CALL random_number(humrelv)
    pref_soil_veg = 2
    !$ACC UPDATE DEVICE(ae_ns, precisol, tr_ns, vegtot, dt_sechiba, precisol_ns, rootsink, check_cwrr, ok_hydrol_arch, vegetmax_soil, humrelv, pref_soil_veg)
  END SUBROUTINE initialization
END MODULE module_global
