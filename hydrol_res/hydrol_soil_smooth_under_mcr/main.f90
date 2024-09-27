
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  LOGICAL, DIMENSION(kjpindex, nstm) :: is_under_mcr, is_under_mcr_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: check, check_cpu
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, njsc, mcr, ji)
  CALL hydrol_soil_smooth_under_mcr(mcr, kjpindex, ins, njsc, is_under_mcr, check)
  mc_cpu = mc
  is_under_mcr_cpu = is_under_mcr
  check_cpu = check
  CALL initialization
  CALL read_dummy(ins, njsc, mcr, ji)
  !$ACC ENTER DATA COPYIN(mcr, njsc, is_under_mcr, check)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_smooth_under_mcr_acc(ji, mcr, kjpindex, ins, njsc, is_under_mcr, check)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(mc, is_under_mcr, check)
  !$ACC EXIT DATA DELETE(mcr, njsc, is_under_mcr, check)
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
  IF (ALL(is_under_mcr .EQV. is_under_mcr_cpu)) THEN
    WRITE(*, *) 'LOGICAL EQV test passed: All elements in is_under_mcr_gpu are equal to is_under_mcr_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'LOGICAL EQV test failed: Not all elements in is_under_mcr_gpu match is_under_mcr_cpu.'
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
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_smooth_under_mcr
  !!
  !>\BRIEF        : Modifies the soil moisture profile to avoid under-residual values,
  !!                then diagnoses the points where such "excess" values remain.
  !!
  !! DESCRIPTION  :
  !! The "excesses" under-residual are corrected from top to bottom, by transfer of excesses
  !! to the lower layers. The reverse transfer is performed to smooth any remaining "excess" in the bottom layer.
  !! If some "excess" remain afterwards, the entire soil profile is at the threshold value (mcs or mcr),
  !! and the remaining "excess" is necessarily concentrated in the top layer.
  !! This allowing diagnosing the flag is_under_mcr.
  !! Eventually, the remaining "excess" is split over the entire profile
  !! 1. We calculate the total SM at the beginning of the routine
  !! 2. Smoothes the profile to avoid negative values of punctual soil moisture
  !! Note that we check that mc > min_sechiba in hydrol_soil
  !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
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
  !_ hydrol_soil_smooth_under_mcr

  SUBROUTINE hydrol_soil_smooth_under_mcr_acc(ji, mcr, kjpindex, ins, njsc, is_under_mcr, check)
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
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex, nstm), INTENT(OUT) :: is_under_mcr
    !! Flag diagnosing under residual soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: excess
    REAL(KIND = r_std) :: excessji
    REAL(KIND = r_std) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std) :: tmcf
    !! total SM at end of routine

    !_ ================================================================================================================================

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci = tmci + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmci = tmci + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
    END IF

      !! 2. Smoothes the profile to avoid negative values of punctual soil moisture

      ! 2.1 smoothing from top to bottom
      DO jsl = 1, nslm - 2
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl + 1) + dz(jsl + 2))
    END DO

    jsl = nslm - 1
    excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
    mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
    mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / dz(jsl + 1)

    jsl = nslm
    excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
    mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
    mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * dz(jsl) / (dz(jsl - 1) + dz(jsl))

      ! 2.2 smoothing from bottom to top
      DO jsl = nslm - 1, 2, - 1
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl - 1) + dz(jsl))
    END DO

    ! 2.3 diagnoses is_under_mcr(ji), and updates the entire profile
    ! excess > 0
    excessji = mask_soiltile(ji, ins) * MAX(mcr(ji) - mc(ji, 1, ins), zero)
    mc(ji, 1, ins) = mc(ji, 1, ins) + excessji
    ! then mc(1)=mcr
    is_under_mcr(ji, ins) = (excessji .GT. min_sechiba)

      ! 2.4 The amount of water corresponding to excess in the top soil layer is redistributed in all soil layers
      ! -excess(ji) * dz(2) / deux donne le deficit total, negatif, en mm
      ! diviser par la profondeur totale en mm donne des delta_mc identiques en chaque couche, en mm
      ! retransformes en delta_mm par couche selon les bonnes eqs (eqs_hydrol.pdf, Eqs 13-15), puis sommes
      ! retourne bien le deficit total en mm
      DO jsl = 1, nslm
      mc(ji, jsl, ins) = mc(ji, jsl, ins) - excessji * dz(2) / (deux * zmaxh * mille)
    END DO
    ! This can lead to mc(jsl) < mcr depending on the value of excess,
      ! but this is no major pb for the diffusion
      ! Yet, we need to prevent evaporation if is_under_mcr

      !! Note that we check that mc > min_sechiba in hydrol_soil

      ! We just make sure that mc remains at 0 where soiltile=0
      DO jsl = 1, nslm
      mc(ji, jsl, ins) = mask_soiltile(ji, ins) * mc(ji, jsl, ins)
    END DO

      !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf = dz(2) * (trois * mc(ji, 1, ins) + mc(ji, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf = tmcf + dz(jsl) * (trois * mc(ji, jsl, ins) + mc(ji, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, ins) + mc(ji, jsl + 1, ins)) / huit
      END DO
      tmcf = tmcf + dz(nslm) * (trois * mc(ji, nslm, ins) + mc(ji, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci since we just redistribute the deficit
      check(ji, ins) = tmcf - tmci
    END IF

  END SUBROUTINE hydrol_soil_smooth_under_mcr_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_smooth_under_mcr
    !!
    !>\BRIEF        : Modifies the soil moisture profile to avoid under-residual values,
    !!                then diagnoses the points where such "excess" values remain.
    !!
    !! DESCRIPTION  :
    !! The "excesses" under-residual are corrected from top to bottom, by transfer of excesses
    !! to the lower layers. The reverse transfer is performed to smooth any remaining "excess" in the bottom layer.
    !! If some "excess" remain afterwards, the entire soil profile is at the threshold value (mcs or mcr),
    !! and the remaining "excess" is necessarily concentrated in the top layer.
    !! This allowing diagnosing the flag is_under_mcr.
    !! Eventually, the remaining "excess" is split over the entire profile
    !! 1. We calculate the total SM at the beginning of the routine
    !! 2. Smoothes the profile to avoid negative values of punctual soil moisture
    !! Note that we check that mc > min_sechiba in hydrol_soil
    !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
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
    !_ hydrol_soil_smooth_under_mcr

    SUBROUTINE hydrol_soil_smooth_under_mcr(mcr, kjpindex, ins, njsc, is_under_mcr, check)

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
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    LOGICAL, DIMENSION(kjpindex, nstm), INTENT(OUT) :: is_under_mcr
    !! Flag diagnosing under residual soil moisture
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(OUT) :: check
    !! delta SM - flux

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: ji, jsl
    REAL(KIND = r_std) :: excess
    REAL(KIND = r_std), DIMENSION(kjpindex) :: excessji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmci
    !! total SM at beginning of routine
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmcf
    !! total SM at end of routine

    !_ ================================================================================================================================

    !! 1. We calculate the total SM at the beginning of the routine
    IF (check_cwrr) THEN
      tmci(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmci(:) = tmci(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmci(:) = tmci(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
    END IF

      !! 2. Smoothes the profile to avoid negative values of punctual soil moisture

      ! 2.1 smoothing from top to bottom
      DO jsl = 1, nslm - 2
      DO ji = 1, kjpindex
        excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
        mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl + 1) + dz(jsl + 2))
      END DO
    END DO

    jsl = nslm - 1
    DO ji = 1, kjpindex
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl + 1, ins) = mc(ji, jsl + 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / dz(jsl + 1)
    END DO

    jsl = nslm
    DO ji = 1, kjpindex
      excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
      mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
      mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * dz(jsl) / (dz(jsl - 1) + dz(jsl))
    END DO

      ! 2.2 smoothing from bottom to top
      DO jsl = nslm - 1, 2, - 1
      DO ji = 1, kjpindex
        excess = MAX(mcr(ji) - mc(ji, jsl, ins), zero)
        mc(ji, jsl, ins) = mc(ji, jsl, ins) + excess
        mc(ji, jsl - 1, ins) = mc(ji, jsl - 1, ins) - excess * (dz(jsl) + dz(jsl + 1)) / (dz(jsl - 1) + dz(jsl))
      END DO
    END DO

      ! 2.3 diagnoses is_under_mcr(ji), and updates the entire profile
      ! excess > 0
      DO ji = 1, kjpindex
      excessji(ji) = mask_soiltile(ji, ins) * MAX(mcr(ji) - mc(ji, 1, ins), zero)
    END DO
    DO ji = 1, kjpindex
      mc(ji, 1, ins) = mc(ji, 1, ins) + excessji(ji)
      ! then mc(1)=mcr
      is_under_mcr(ji, ins) = (excessji(ji) .GT. min_sechiba)
    END DO

      ! 2.4 The amount of water corresponding to excess in the top soil layer is redistributed in all soil layers
      ! -excess(ji) * dz(2) / deux donne le deficit total, negatif, en mm
      ! diviser par la profondeur totale en mm donne des delta_mc identiques en chaque couche, en mm
      ! retransformes en delta_mm par couche selon les bonnes eqs (eqs_hydrol.pdf, Eqs 13-15), puis sommes
      ! retourne bien le deficit total en mm
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        mc(ji, jsl, ins) = mc(ji, jsl, ins) - excessji(ji) * dz(2) / (deux * zmaxh * mille)
      END DO
    END DO
    ! This can lead to mc(jsl) < mcr depending on the value of excess,
      ! but this is no major pb for the diffusion
      ! Yet, we need to prevent evaporation if is_under_mcr

      !! Note that we check that mc > min_sechiba in hydrol_soil

      ! We just make sure that mc remains at 0 where soiltile=0
      DO jsl = 1, nslm
      DO ji = 1, kjpindex
        mc(ji, jsl, ins) = mask_soiltile(ji, ins) * mc(ji, jsl, ins)
      END DO
    END DO

      !! 3. For water conservation check, We calculate the total SM at the beginning of the routine,
      !!    and export the difference with the flux
      IF (check_cwrr) THEN
      tmcf(:) = dz(2) * (trois * mc(:, 1, ins) + mc(:, 2, ins)) / huit
      DO jsl = 2, nslm - 1
        tmcf(:) = tmcf(:) + dz(jsl) * (trois * mc(:, jsl, ins) + mc(:, jsl - 1, ins)) / huit + dz(jsl + 1) * (trois * mc(:, jsl, ins) + mc(:, jsl + 1, ins)) / huit
      END DO
      tmcf(:) = tmcf(:) + dz(nslm) * (trois * mc(:, nslm, ins) + mc(:, nslm - 1, ins)) / huit
      ! Normally, tcmf=tmci since we just redistribute the deficit
      check(:, ins) = tmcf(:) - tmci(:)
    END IF

  END SUBROUTINE hydrol_soil_smooth_under_mcr
  SUBROUTINE read_dummy(ins, njsc, mcr, ji)
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    njsc = 2
    CALL random_number(mcr)
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
