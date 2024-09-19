
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mclint
  REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_top
  INTEGER(KIND = i_std) :: ji
  INTEGER(KIND = i_std) :: error_flag_hydrol_diag_soil_flux_1
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, mclint, flux_top, ji, error_flag_hydrol_diag_soil_flux_1)
  CALL hydrol_diag_soil_flux(kjpindex, ins, mclint, flux_top)
  check_top_ns_cpu = check_top_ns
  qflux_ns_cpu = qflux_ns
  CALL initialization
  CALL read_dummy(ins, mclint, flux_top, ji, error_flag_hydrol_diag_soil_flux_1)
  error_flag_hydrol_diag_soil_flux_1 = 0
  !$ACC ENTER DATA COPYIN(mclint, flux_top)
  !$ACC PARALLEL LOOP INDEPENDENT REDUCTION(+:error_flag_hydrol_diag_soil_flux_1)
  DO ji = 1, kjpindex
    CALL hydrol_diag_soil_flux_acc(error_flag_hydrol_diag_soil_flux_1, ji, kjpindex, ins, mclint, flux_top)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(check_top_ns, qflux_ns)
  !$ACC EXIT DATA DELETE(mclint, flux_top)
  IF (ALL(check_top_ns .EQ. check_top_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in check_top_ns_gpu are equal to check_top_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in check_top_ns_gpu do not match check_top_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(check_top_ns - check_top_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of check_top_ns_gpu:', MINVAL(check_top_ns), MAXVAL(check_top_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of check_top_ns_cpu:', MINVAL(check_top_ns_cpu), MAXVAL(check_top_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(qflux_ns .EQ. qflux_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in qflux_ns_gpu are equal to qflux_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in qflux_ns_gpu do not match qflux_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(qflux_ns - qflux_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qflux_ns_gpu:', MINVAL(qflux_ns), MAXVAL(qflux_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qflux_ns_cpu:', MINVAL(qflux_ns_cpu), MAXVAL(qflux_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (error_flag_hydrol_diag_soil_flux_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_diag_soil_flux, error_flag_hydrol_diag_soil_flux_1 is > 0 :', error_flag_hydrol_diag_soil_flux_1
    CALL ipslerr_p(1, 'hydrol_diag_soil_flux', 'NOTE:', 'Problem in the water balance, qflux_ns computation', '')
  END IF
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_diag_soil_flux
  !!
  !>\BRIEF        : This subroutine diagnoses the vertical liquid water fluxes between the
  !!                different soil layers, based on each layer water budget. It also checks the
  !!                corresponding water conservation (during redistribution).
  !!
  !! DESCRIPTION  :
  !! 1. Initialize qflux_ns from the bottom, with dr_ns
  !! 2. Between layer nslm and nslm-1, by means of water budget knowing mc changes and flux at the lowest interface
  !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
  !! 4. Water balance verification: pursuing upward water budget, the flux at the surface should equal -flux_top
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne to fit hydrol_soil
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================

  SUBROUTINE hydrol_diag_soil_flux_acc(error_flag_hydrol_diag_soil_flux_1, ji, kjpindex, ins, mclint, flux_top)
    !$ACC ROUTINE SEQ
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_diag_soil_flux_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: mclint
    !! mc values at the beginning of the time step
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_top
    !! Exfiltration (bare soil evaporation minus infiltration)

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables
    REAL(KIND = r_std) :: check_temp
    !! Diagnosed flux at soil surface, should equal -flux_top
    INTEGER(KIND = i_std) :: jsl

    !_ ================================================================================================================================

    !- Compute the diffusion flux at every level from bottom to top (using mcl,mclint, and sink values)

    !! 1. Initialize qflux_ns from the bottom, with dr_ns
    jsl = nslm
    qflux_ns(ji, jsl, ins) = dr_ns(ji, ins)
    !! 2. Between layer nslm and nslm-1, by means of water budget
    !!    knowing mc changes and flux at the lowest interface
    !     qflux_ns is downward
    jsl = nslm - 1
    qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins)

      !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
      ! Here, qflux_ns(ji,1,ins) is the downward flux between the top soil layer and the 2nd one
      DO jsl = nslm - 2, 1, - 1
      qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins) + (dz(jsl + 2) / huit) * (trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1) + mcl(ji, jsl + 2, ins) - mclint(ji, jsl + 2))
    END DO

    !! 4. Water balance verification: pursuing upward water budget, the flux at the surface (check_temp)
    !! should equal -flux_top

    check_temp = qflux_ns(ji, 1, ins) + (dz(2) / huit) * (trois * (mcl(ji, 1, ins) - mclint(ji, 1)) + (mcl(ji, 2, ins) - mclint(ji, 2))) + rootsink(ji, 1, ins)
    ! flux_top is positive when upward, while check_temp is positive when downward
    check_top_ns(ji, ins) = flux_top(ji) + check_temp

      IF (ABS(check_top_ns(ji, ins)) / dt_sechiba .GT. min_sechiba) THEN
      ! Diagnosed (check_temp) and imposed (flux_top) differ by more than 1.e-8 mm/s
      error_flag_hydrol_diag_soil_flux_1 = error_flag_hydrol_diag_soil_flux_1 + 1
    END IF

  END SUBROUTINE hydrol_diag_soil_flux_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_diag_soil_flux
    !!
    !>\BRIEF        : This subroutine diagnoses the vertical liquid water fluxes between the
    !!                different soil layers, based on each layer water budget. It also checks the
    !!                corresponding water conservation (during redistribution).
    !!
    !! DESCRIPTION  :
    !! 1. Initialize qflux_ns from the bottom, with dr_ns
    !! 2. Between layer nslm and nslm-1, by means of water budget knowing mc changes and flux at the lowest interface
    !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
    !! 4. Water balance verification: pursuing upward water budget, the flux at the surface should equal -flux_top
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne to fit hydrol_soil
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================

    SUBROUTINE hydrol_diag_soil_flux(kjpindex, ins, mclint, flux_top)
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: mclint
    !! mc values at the beginning of the time step
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_top
    !! Exfiltration (bare soil evaporation minus infiltration)

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables
    REAL(KIND = r_std), DIMENSION(kjpindex) :: check_temp
    !! Diagnosed flux at soil surface, should equal -flux_top
    INTEGER(KIND = i_std) :: jsl, ji

    !_ ================================================================================================================================

    !- Compute the diffusion flux at every level from bottom to top (using mcl,mclint, and sink values)
    DO ji = 1, kjpindex

      !! 1. Initialize qflux_ns from the bottom, with dr_ns
      jsl = nslm
      qflux_ns(ji, jsl, ins) = dr_ns(ji, ins)
      !! 2. Between layer nslm and nslm-1, by means of water budget
      !!    knowing mc changes and flux at the lowest interface
      !     qflux_ns is downward
      jsl = nslm - 1
      qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins)
    END DO

      !! 3. We go up, and deduct qflux_ns(1:nslm-2), still by means of water budget
      ! Here, qflux_ns(ji,1,ins) is the downward flux between the top soil layer and the 2nd one
      DO jsl = nslm - 2, 1, - 1
      DO ji = 1, kjpindex
        qflux_ns(ji, jsl, ins) = qflux_ns(ji, jsl + 1, ins) + (mcl(ji, jsl, ins) - mclint(ji, jsl) + trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1)) * (dz(jsl + 1) / huit) + rootsink(ji, jsl + 1, ins) + (dz(jsl + 2) / huit) * (trois * mcl(ji, jsl + 1, ins) - trois * mclint(ji, jsl + 1) + mcl(ji, jsl + 2, ins) - mclint(ji, jsl + 2))
      END DO
    END DO

      !! 4. Water balance verification: pursuing upward water budget, the flux at the surface (check_temp)
      !! should equal -flux_top
      DO ji = 1, kjpindex

      check_temp(ji) = qflux_ns(ji, 1, ins) + (dz(2) / huit) * (trois * (mcl(ji, 1, ins) - mclint(ji, 1)) + (mcl(ji, 2, ins) - mclint(ji, 2))) + rootsink(ji, 1, ins)
      ! flux_top is positive when upward, while check_temp is positive when downward
      check_top_ns(ji, ins) = flux_top(ji) + check_temp(ji)

        IF (ABS(check_top_ns(ji, ins)) / dt_sechiba .GT. min_sechiba) THEN
        ! Diagnosed (check_temp) and imposed (flux_top) differ by more than 1.e-8 mm/s
        WRITE(numout, *) 'Problem in the water balance, qflux_ns computation, surface fluxes', flux_top(ji), check_temp(ji)
        WRITE(numout, *) 'Diagnosed and imposed fluxes differ by more than 1.e-8 mm/s: ', check_top_ns(ji, ins)
        WRITE(numout, *) 'ji', ji, 'jsl', jsl, 'ins', ins
        WRITE(numout, *) 'mclint', mclint(ji, :)
        WRITE(numout, *) 'mcl', mcl(ji, :, ins)
        WRITE(numout, *) 'rootsink', rootsink(ji, 1, ins)
        CALL ipslerr_p(1, 'hydrol_diag_soil_flux', 'NOTE:', 'Problem in the water balance, qflux_ns computation', '')
      END IF
    END DO

  END SUBROUTINE hydrol_diag_soil_flux
  SUBROUTINE read_dummy(ins, mclint, flux_top, ji, error_flag_hydrol_diag_soil_flux_1)
    INTEGER(KIND = i_std) :: error_flag_hydrol_diag_soil_flux_1
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_top
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mclint
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    CALL random_number(mclint)
    CALL random_number(flux_top)
    ji = 2
    error_flag_hydrol_diag_soil_flux_1 = 2
  END SUBROUTINE read_dummy
END PROGRAM main
