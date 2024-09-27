
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
  REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
  REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
  REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_infilt
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check, check_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: ru_infilt, ru_infilt_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: qinfilt_ns, qinfilt_ns_cpu
  INTEGER(KIND = i_std) :: ji
  INTEGER(KIND = i_std) :: error_flag_hydrol_soil_infilt_1
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, njsc, ks, nvan, avan, mcr, mcs, mcfc, mcw, flux_infilt, stempdiag, ji, error_flag_hydrol_soil_infilt_1)
  CALL hydrol_soil_infilt(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
  mc_cpu = mc
  check_cpu = check
  ru_infilt_cpu = ru_infilt
  qinfilt_ns_cpu = qinfilt_ns
  CALL initialization
  CALL read_dummy(ins, njsc, ks, nvan, avan, mcr, mcs, mcfc, mcw, flux_infilt, stempdiag, ji, error_flag_hydrol_soil_infilt_1)
  error_flag_hydrol_soil_infilt_1 = 0
  !$ACC ENTER DATA COPYIN(ks, nvan, avan, mcr, mcs, mcfc, mcw, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
  !$ACC PARALLEL LOOP INDEPENDENT REDUCTION(+:error_flag_hydrol_soil_infilt_1)
  DO ji = 1, kjpindex
    CALL hydrol_soil_infilt_acc(error_flag_hydrol_soil_infilt_1, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(mc, check, ru_infilt, qinfilt_ns)
  !$ACC EXIT DATA DELETE(ks, nvan, avan, mcr, mcs, mcfc, mcw, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
  IF (ALL(mc .EQ. mc_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mc_gpu are equal to mc_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mc_gpu do not match mc_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mc - mc_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_gpu:', MINVAL(mc), MAXVAL(mc)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_cpu:', MINVAL(mc_cpu), MAXVAL(mc_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(check .EQ. check_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in check_gpu are equal to check_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in check_gpu do not match check_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(check - check_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of check_gpu:', MINVAL(check), MAXVAL(check)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of check_cpu:', MINVAL(check_cpu), MAXVAL(check_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(ru_infilt .EQ. ru_infilt_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ru_infilt_gpu are equal to ru_infilt_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ru_infilt_gpu do not match ru_infilt_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ru_infilt - ru_infilt_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ru_infilt_gpu:', MINVAL(ru_infilt), MAXVAL(ru_infilt)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ru_infilt_cpu:', MINVAL(ru_infilt_cpu), MAXVAL(ru_infilt_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(qinfilt_ns .EQ. qinfilt_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in qinfilt_ns_gpu are equal to qinfilt_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in qinfilt_ns_gpu do not match qinfilt_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(qinfilt_ns - qinfilt_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qinfilt_ns_gpu:', MINVAL(qinfilt_ns), MAXVAL(qinfilt_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qinfilt_ns_cpu:', MINVAL(qinfilt_ns_cpu), MAXVAL(qinfilt_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (error_flag_hydrol_soil_infilt_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_soil_infilt, error_flag_hydrol_soil_infilt_1 is > 0 :', error_flag_hydrol_soil_infilt_1
    CALL ipslerr_p(3, 'hydrol_soil_infilt', 'We will STOP now.', 'Error in calculation of infilt tot', '')
  END IF
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_infilt
  !!
  !>\BRIEF        Infiltration
  !!
  !! DESCRIPTION  :
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. Infiltration process
  !! 2.1 Initialization of time counter and infiltration rate
  !! 2.2 Infiltration layer by layer, accounting for an exponential law for subgrid variability
  !! 2.3 Resulting infiltration and surface runoff
  !! 3. For water conservation check, we calculate the total SM at the beginning of the routine,
  !!    and export the difference with the flux
  !! 5. Local verification
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !! Adding checks and interactions variables with hydrol_soil, but the processes are unchanged
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================
  !_ hydrol_soil_infilt

  SUBROUTINE hydrol_soil_infilt_acc(error_flag_hydrol_soil_infilt_1, ji, ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)
    !$ACC ROUTINE SEQ

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! GLOBAL (in or inout)
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_soil_infilt_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell
    !!  (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: ru_infilt
    !! Surface runoff from soil_infilt (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: qinfilt_ns
    !! Effective infiltration flux (mm/dt_sechiba)

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    !! Indices
    REAL(KIND = r_std) :: wat_inf_pot
    !! infiltrable water in the layer
    REAL(KIND = r_std) :: wat_inf
    !! infiltrated water in the layer
    REAL(KIND = r_std) :: dt_tmp
    !! time remaining before the end of the time step
    REAL(KIND = r_std) :: dt_inf
    !! the time it takes to complete the infiltration in the
    !! layer
    REAL(KIND = r_std) :: k_m
    !! the mean conductivity used for the saturated front
    REAL(KIND = r_std) :: infilt_tmp
    !! infiltration rate for the considered layer
    REAL(KIND = r_std) :: infilt_tot
    !! total infiltration
    REAL(KIND = r_std) :: flux_tmp
    !! rate at which precip hits the ground

    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine (kg/m2)
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine (kg/m2)


    !_ ================================================================================================================================

    ! If data (or coupling with GCM) was available, a parameterization for subgrid rainfall could be performed

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

    !! 2. Infiltration process

    !! 2.1 Initialization

    !! First we fill up the first layer (about 1mm) without any resistance and quasi-immediately
    wat_inf_pot = MAX((mcs(ji) - mc(ji, 1, ins)) * dz(2) / deux, zero)
    wat_inf = MIN(wat_inf_pot, flux_infilt(ji))
    mc(ji, 1, ins) = mc(ji, 1, ins) + wat_inf * deux / dz(2)
    !

    !! Initialize a countdown for infiltration during the time-step and the value of potential runoff
    dt_tmp = dt_sechiba / one_day
    infilt_tot = wat_inf
    !! Compute the rate at which water will try to infiltrate each layer
    ! flux_temp is converted here to the same unit as k_m
    flux_tmp = (flux_infilt(ji) - wat_inf) / dt_tmp

      !! 2.2 Infiltration layer by layer
      DO jsl = 2, nslm - 1
      !! Infiltrability of each layer if under a saturated one
      ! This is computed by an simple arithmetic average because
      ! the time step (30min) is not appropriate for a geometric average (advised by Haverkamp and Vauclin)
      k_m = (k(ji, jsl) + ks(ji) * kfact(jsl - 1, ji) * kfact_root(ji, jsl, ins)) / deux

        IF (ok_freeze_cwrr) THEN
        IF (stempdiag(ji, jsl) .LT. ZeroCelsius) THEN
          k_m = k(ji, jsl)
        END IF
      END IF

      !! We compute the mean rate at which water actually infiltrate:
      ! Subgrid: Exponential distribution of k around k_m, but average p directly used
      ! See d'Orgeval 2006, p 78, but it's not fully clear to me (AD16***)
      infilt_tmp = k_m * (un - EXP(- flux_tmp / k_m))

      !! From which we deduce the time it takes to fill up the layer or to end the time step...
      wat_inf_pot = MAX((mcs(ji) - mc(ji, jsl, ins)) * (dz(jsl) + dz(jsl + 1)) / deux, zero)
      IF (infilt_tmp > min_sechiba) THEN
        dt_inf = MIN(wat_inf_pot / infilt_tmp, dt_tmp)
        ! The water infiltration TIME has to limited by what is still available for infiltration.
          IF (dt_inf * infilt_tmp > flux_infilt(ji) - infilt_tot) THEN
          dt_inf = MAX(flux_infilt(ji) - infilt_tot, zero) / infilt_tmp
        END IF
      ELSE
        dt_inf = dt_tmp
      END IF

      !! The water enters in the layer
      wat_inf = dt_inf * infilt_tmp
      ! bviously the moisture content
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + wat_inf * deux / (dz(jsl) + dz(jsl + 1))
      ! the time remaining before the next time step
      dt_tmp = dt_tmp - dt_inf
      ! and finally the infilt_tot (which is just used to check if there is a problem, below)
      infilt_tot = infilt_tot + infilt_tmp * dt_inf
    END DO

    !! 2.3 Resulting infiltration and surface runoff
    ru_infilt(ji, ins) = flux_infilt(ji) - infilt_tot
    qinfilt_ns(ji, ins) = infilt_tot

      !! 3. For water conservation check: we calculate the total SM at the beginning of the routine
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci+infilt_tot
      check(ji, ins) = tmcf - (tmci + infilt_tot)
    END IF

      !! 5. Local verification
      IF (infilt_tot .LT. - min_sechiba .OR. infilt_tot .GT. flux_infilt(ji) + min_sechiba) THEN
      error_flag_hydrol_soil_infilt_1 = error_flag_hydrol_soil_infilt_1 + 1
    END IF

  END SUBROUTINE hydrol_soil_infilt_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_infilt
    !!
    !>\BRIEF        Infiltration
    !!
    !! DESCRIPTION  :
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. Infiltration process
    !! 2.1 Initialization of time counter and infiltration rate
    !! 2.2 Infiltration layer by layer, accounting for an exponential law for subgrid variability
    !! 2.3 Resulting infiltration and surface runoff
    !! 3. For water conservation check, we calculate the total SM at the beginning of the routine,
    !!    and export the difference with the flux
    !! 5. Local verification
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !! Adding checks and interactions variables with hydrol_soil, but the processes are unchanged
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================
    !_ hydrol_soil_infilt

    SUBROUTINE hydrol_soil_infilt(ks, nvan, avan, mcr, mcs, mcfc, mcw, kjpindex, ins, njsc, flux_infilt, stempdiag, qinfilt_ns, ru_infilt, check)

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    ! GLOBAL (in or inout)
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell
    !!  (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: ks
    !! Hydraulic conductivity at saturation (mm {-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcfc
    !! Volumetric water content at field capacity (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcw
    !! Volumetric water content at wilting point (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: flux_infilt
    !! Water to infiltrate
    !!  @tex $(kg m^{-2})$ @endtex
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: ru_infilt
    !! Surface runoff from soil_infilt (mm/dt_sechiba)
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: qinfilt_ns
    !! Effective infiltration flux (mm/dt_sechiba)

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: ji, jsl
    !! Indices
    REAL(KIND = r_std), DIMENSION(kjpindex) :: wat_inf_pot
    !! infiltrable water in the layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: wat_inf
    !! infiltrated water in the layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dt_tmp
    !! time remaining before the end of the time step
    REAL(KIND = r_std), DIMENSION(kjpindex) :: dt_inf
    !! the time it takes to complete the infiltration in the
    !! layer
    REAL(KIND = r_std) :: k_m
    !! the mean conductivity used for the saturated front
    REAL(KIND = r_std), DIMENSION(kjpindex) :: infilt_tmp
    !! infiltration rate for the considered layer
    REAL(KIND = r_std), DIMENSION(kjpindex) :: infilt_tot
    !! total infiltration
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_tmp
    !! rate at which precip hits the ground

    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine (kg/m2)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine (kg/m2)


    !_ ================================================================================================================================

    ! If data (or coupling with GCM) was available, a parameterization for subgrid rainfall could be performed

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. Infiltration process

      !! 2.1 Initialization

      DO ji = 1, kjpindex
      !! First we fill up the first layer (about 1mm) without any resistance and quasi-immediately
      wat_inf_pot(ji) = MAX((mcs(ji) - mc(ji, 1, ins)) * dz(2) / deux, zero)
      wat_inf(ji) = MIN(wat_inf_pot(ji), flux_infilt(ji))
      mc(ji, 1, ins) = mc(ji, 1, ins) + wat_inf(ji) * deux / dz(2)
      !
    END DO

    !! Initialize a countdown for infiltration during the time-step and the value of potential runoff
    dt_tmp(:) = dt_sechiba / one_day
    infilt_tot(:) = wat_inf(:)
    !! Compute the rate at which water will try to infiltrate each layer
    ! flux_temp is converted here to the same unit as k_m
    flux_tmp(:) = (flux_infilt(:) - wat_inf(:)) / dt_tmp(:)

      !! 2.2 Infiltration layer by layer
      DO jsl = 2, nslm - 1
      DO ji = 1, kjpindex
        !! Infiltrability of each layer if under a saturated one
        ! This is computed by an simple arithmetic average because
        ! the time step (30min) is not appropriate for a geometric average (advised by Haverkamp and Vauclin)
        k_m = (k(ji, jsl) + ks(ji) * kfact(jsl - 1, ji) * kfact_root(ji, jsl, ins)) / deux

          IF (ok_freeze_cwrr) THEN
          IF (stempdiag(ji, jsl) .LT. ZeroCelsius) THEN
            k_m = k(ji, jsl)
          END IF
        END IF

        !! We compute the mean rate at which water actually infiltrate:
        ! Subgrid: Exponential distribution of k around k_m, but average p directly used
        ! See d'Orgeval 2006, p 78, but it's not fully clear to me (AD16***)
        infilt_tmp(ji) = k_m * (un - EXP(- flux_tmp(ji) / k_m))

        !! From which we deduce the time it takes to fill up the layer or to end the time step...
        wat_inf_pot(ji) = MAX((mcs(ji) - mc(ji, jsl, ins)) * (dz(jsl) + dz(jsl + 1)) / deux, zero)
        IF (infilt_tmp(ji) > min_sechiba) THEN
          dt_inf(ji) = MIN(wat_inf_pot(ji) / infilt_tmp(ji), dt_tmp(ji))
          ! The water infiltration TIME has to limited by what is still available for infiltration.
            IF (dt_inf(ji) * infilt_tmp(ji) > flux_infilt(ji) - infilt_tot(ji)) THEN
            dt_inf(ji) = MAX(flux_infilt(ji) - infilt_tot(ji), zero) / infilt_tmp(ji)
          END IF
        ELSE
          dt_inf(ji) = dt_tmp(ji)
        END IF

        !! The water enters in the layer
        wat_inf(ji) = dt_inf(ji) * infilt_tmp(ji)
        ! bviously the moisture content
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + wat_inf(ji) * deux / (dz(jsl) + dz(jsl + 1))
        ! the time remaining before the next time step
        dt_tmp(ji) = dt_tmp(ji) - dt_inf(ji)
        ! and finally the infilt_tot (which is just used to check if there is a problem, below)
        infilt_tot(ji) = infilt_tot(ji) + infilt_tmp(ji) * dt_inf(ji)
      END DO
    END DO

    !! 2.3 Resulting infiltration and surface runoff
    ru_infilt(:, ins) = flux_infilt(:) - infilt_tot(:)
    qinfilt_ns(:, ins) = infilt_tot(:)

      !! 3. For water conservation check: we calculate the total SM at the beginning of the routine
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci+infilt_tot
      check(:, ins) = tmcf(:) - (tmci(:) + infilt_tot(:))
    END IF

      !! 5. Local verification
      DO ji = 1, kjpindex
      IF (infilt_tot(ji) .LT. - min_sechiba .OR. infilt_tot(ji) .GT. flux_infilt(ji) + min_sechiba) THEN
        WRITE(numout, *) 'Error in the calculation of infilt tot', infilt_tot(ji)
        WRITE(numout, *) 'k, ji, jst, mc', k(ji, 1 : 2), ji, ins, mc(ji, 1, ins)
        CALL ipslerr_p(3, 'hydrol_soil_infilt', 'We will STOP now.', 'Error in calculation of infilt tot', '')
      END IF
    END DO

  END SUBROUTINE hydrol_soil_infilt
  SUBROUTINE read_dummy(ins, njsc, ks, nvan, avan, mcr, mcs, mcfc, mcw, flux_infilt, stempdiag, ji, error_flag_hydrol_soil_infilt_1)
    INTEGER(KIND = i_std) :: error_flag_hydrol_soil_infilt_1
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
    REAL(KIND = r_std), DIMENSION(kjpindex) :: flux_infilt
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcw
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcfc
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
    REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
    REAL(KIND = r_std), DIMENSION(kjpindex) :: ks
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    njsc = 2
    CALL random_number(ks)
    CALL random_number(nvan)
    CALL random_number(avan)
    CALL random_number(mcr)
    CALL random_number(mcs)
    CALL random_number(mcfc)
    CALL random_number(mcw)
    CALL random_number(flux_infilt)
    CALL random_number(stempdiag)
    ji = 2
    error_flag_hydrol_soil_infilt_1 = 2
  END SUBROUTINE read_dummy
END PROGRAM main
