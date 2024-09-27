
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  LOGICAL, DIMENSION(kjpindex) :: is_over_mcs, is_over_mcs_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check, check_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: rudr_corr, rudr_corr_cpu
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, njsc, mcs, rudr_corr, ji)
  CALL hydrol_soil_smooth_over_mcs2(mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)
  mc_cpu = mc
  is_over_mcs_cpu = is_over_mcs
  check_cpu = check
  rudr_corr_cpu = rudr_corr
  CALL initialization
  CALL read_dummy(ins, njsc, mcs, rudr_corr, ji)
  !$ACC ENTER DATA COPYIN(mcs, njsc, is_over_mcs, rudr_corr, check)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_smooth_over_mcs2_acc(ji, mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(mc, is_over_mcs, check, rudr_corr)
  !$ACC EXIT DATA DELETE(mcs, njsc, is_over_mcs, rudr_corr, check)
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
  IF (ALL(is_over_mcs .EQV. is_over_mcs_cpu)) THEN
    WRITE(*, *) 'LOGICAL EQV test passed: All elements in is_over_mcs_gpu are equal to is_over_mcs_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'LOGICAL EQV test failed: Not all elements in is_over_mcs_gpu match is_over_mcs_cpu.'
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
  IF (ALL(rudr_corr .EQ. rudr_corr_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in rudr_corr_gpu are equal to rudr_corr_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in rudr_corr_gpu do not match rudr_corr_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(rudr_corr - rudr_corr_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rudr_corr_gpu:', MINVAL(rudr_corr), MAXVAL(rudr_corr)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rudr_corr_cpu:', MINVAL(rudr_corr_cpu), MAXVAL(rudr_corr_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS

  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_smooth_over_mcs2
  !!
  !>\BRIEF        : Modifies the soil moisture profile to avoid over-saturation values,
  !!                by putting the excess in ru_ns
  !!                Thus, no point remain where such "excess" values remain (is_over_mcs becomes useless)
  !!
  !! DESCRIPTION  :
  !! The "excesses" over-saturation are corrected, by directly discarding the excess as rudr_corr,
  !! to be added to ru_ns or dr_nsrunoff (via rudr_corr).
  !! Therefore, there is no more smoothing, and this helps preventing the saturation of too many layers,
  !! which leads to numerical errors with tridiag.
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. In case of over-saturation, we directly eliminate the excess via rudr_corr
  !!    The calculation of the adjustement flux needs to account for nodes n-1 and n+1.
  !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
  !!    and export the difference with the flux
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================
  !_ hydrol_soil_smooth_over_mcs2

  SUBROUTINE hydrol_soil_smooth_over_mcs2_acc(ji, mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)
    !$ACC ROUTINE SEQ

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex), INTENT(OUT) :: is_over_mcs
    !! Flag diagnosing over saturated soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(INOUT) :: rudr_corr
    !! Surface runoff produced to correct excess (mm/dtstep)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std), DIMENSION(nslm) :: excess
    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine

    !_ ================================================================================================================================
    !-

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

      !! 2. In case of over-saturation, we don't do any smoothing,
      !! but directly eliminate the excess as runoff (via rudr_corr)
      !    we correct the calculation of the adjustement flux, which needs to account for nodes n-1 and n+1
      !    for the calculation to remain simple and accurate, we directly drain all the oversaturated mc,
      !    without transfering to lower layers

      !! 2.1 thresholding from top to bottom, with excess defined along jsl
      DO jsl = 1, nslm
      excess(jsl) = MAX(mc(ji, jsl, ins) - mcs(ji), zero)
      ! >=0
      mc(ji, jsl, ins) = mc(ji, jsl, ins) - excess(jsl)
      ! here mc either does not change or decreases
    END DO

    !! 2.2 To ensure conservation, this needs to be balanced by additional drainage (in kg/m2/dt)
    rudr_corr(ji, ins) = dz(2) * (trois * excess(1) + excess(2)) / huit
    ! top layer = initialisation
      DO jsl = 2, nslm - 1
      ! intermediate layers
      rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(jsl) * (trois * excess(jsl) + excess(jsl - 1)) / huit + dz(jsl + 1) * (trois * excess(jsl) + excess(jsl + 1)) / huit
    END DO
    rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(nslm) * (trois * excess(nslm) + excess(nslm - 1)) / huit
    ! bottom layer
    is_over_mcs(ji) = .FALSE.

      !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux

      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci-rudr_corr
      check(ji, ins) = tmcf - (tmci - rudr_corr(ji, ins))
    END IF

  END SUBROUTINE hydrol_soil_smooth_over_mcs2_acc

    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_smooth_over_mcs2
    !!
    !>\BRIEF        : Modifies the soil moisture profile to avoid over-saturation values,
    !!                by putting the excess in ru_ns
    !!                Thus, no point remain where such "excess" values remain (is_over_mcs becomes useless)
    !!
    !! DESCRIPTION  :
    !! The "excesses" over-saturation are corrected, by directly discarding the excess as rudr_corr,
    !! to be added to ru_ns or dr_nsrunoff (via rudr_corr).
    !! Therefore, there is no more smoothing, and this helps preventing the saturation of too many layers,
    !! which leads to numerical errors with tridiag.
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. In case of over-saturation, we directly eliminate the excess via rudr_corr
    !!    The calculation of the adjustement flux needs to account for nodes n-1 and n+1.
    !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
    !!    and export the difference with the flux
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================
    !_ hydrol_soil_smooth_over_mcs2

    SUBROUTINE hydrol_soil_smooth_over_mcs2(mcs, kjpindex, ins, njsc, is_over_mcs, rudr_corr, check)

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Soiltile index (1-nstm, unitless)
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in grid cell
    !! (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex), INTENT(OUT) :: is_over_mcs
    !! Flag diagnosing over saturated soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(INOUT) :: rudr_corr
    !! Surface runoff produced to correct excess (mm/dtstep)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: ji, jsl
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: excess
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine

    !_ ================================================================================================================================
    !-

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. In case of over-saturation, we don't do any smoothing,
      !! but directly eliminate the excess as runoff (via rudr_corr)
      !    we correct the calculation of the adjustement flux, which needs to account for nodes n-1 and n+1
      !    for the calculation to remain simple and accurate, we directly drain all the oversaturated mc,
      !    without transfering to lower layers

      !! 2.1 thresholding from top to bottom, with excess defined along jsl
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        excess(ji, jsl) = MAX(mc(ji, jsl, ins) - mcs(ji), zero)
        ! >=0
        mc(ji, jsl, ins) = mc(ji, jsl, ins) - excess(ji, jsl)
        ! here mc either does not change or decreases
      END DO
    END DO

      !! 2.2 To ensure conservation, this needs to be balanced by additional drainage (in kg/m2/dt)
      DO ji = 1, kjpindex
      rudr_corr(ji, ins) = dz(2) * (trois * excess(ji, 1) + excess(ji, 2)) / huit
      ! top layer = initialisation
    END DO
    DO jsl = 2, nslm - 1
      ! intermediate layers
        DO ji = 1, kjpindex
        rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(jsl) * (trois * excess(ji, jsl) + excess(ji, jsl - 1)) / huit + dz(jsl + 1) * (trois * excess(ji, jsl) + excess(ji, jsl + 1)) / huit
      END DO
    END DO
    DO ji = 1, kjpindex
      rudr_corr(ji, ins) = rudr_corr(ji, ins) + dz(nslm) * (trois * excess(ji, nslm) + excess(ji, nslm - 1)) / huit
      ! bottom layer
      is_over_mcs(ji) = .FALSE.
    END DO

      !! 3. For water conservation checks, we calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux

      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci-rudr_corr
      check(:, ins) = tmcf(:) - (tmci(:) - rudr_corr(:, ins))
    END IF

  END SUBROUTINE hydrol_soil_smooth_over_mcs2
  SUBROUTINE read_dummy(ins, njsc, mcs, rudr_corr, ji)
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: rudr_corr
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    njsc = 2
    CALL random_number(mcs)
    CALL random_number(rudr_corr)
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
