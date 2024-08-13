
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
  REAL(KIND = r_std), DIMENSION(kjpindex) :: drain_upd, drain_upd_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: runoff_upd, runoff_upd_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: qsintveg, qsintveg_cpu
  INTEGER(KIND = i_std) :: ji
  INTEGER(KIND = i_std) :: error_flag_hydrol_tmc_update_1
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(veget_max, soiltile, qsintveg, ji, error_flag_hydrol_tmc_update_1)
  CALL hydrol_tmc_update(kjpindex, veget_max, soiltile, qsintveg, drain_upd, runoff_upd)
  humtot_cpu = humtot
  mc_cpu = mc
  water2infilt_cpu = water2infilt
  tmc_cpu = tmc
  resdist_cpu = resdist
  drain_upd_cpu = drain_upd
  runoff_upd_cpu = runoff_upd
  qsintveg_cpu = qsintveg
  CALL initialization
  CALL read_dummy(veget_max, soiltile, qsintveg, ji, error_flag_hydrol_tmc_update_1)
  error_flag_hydrol_tmc_update_1 = 0
  !$ACC ENTER DATA COPYIN(veget_max, soiltile, qsintveg, drain_upd, runoff_upd)
  !$ACC PARALLEL LOOP INDEPENDENT REDUCTION(+:error_flag_hydrol_tmc_update_1)
  DO ji = 1, kjpindex
    CALL hydrol_tmc_update_acc(error_flag_hydrol_tmc_update_1, ji, kjpindex, veget_max, soiltile, qsintveg, drain_upd, runoff_upd)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(humtot, mc, water2infilt, tmc, resdist, drain_upd, runoff_upd, qsintveg)
  !$ACC EXIT DATA DELETE(veget_max, soiltile, qsintveg, drain_upd, runoff_upd)
  IF (ALL(humtot .EQ. humtot_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in humtot_gpu are equal to humtot_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in humtot_gpu do not match humtot_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(humtot - humtot_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humtot_gpu:', MINVAL(humtot), MAXVAL(humtot)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of humtot_cpu:', MINVAL(humtot_cpu), MAXVAL(humtot_cpu)
    WRITE(*, *) ''
  END IF
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
  IF (ALL(water2infilt .EQ. water2infilt_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in water2infilt_gpu are equal to water2infilt_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in water2infilt_gpu do not match water2infilt_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(water2infilt - water2infilt_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of water2infilt_gpu:', MINVAL(water2infilt), MAXVAL(water2infilt)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of water2infilt_cpu:', MINVAL(water2infilt_cpu), MAXVAL(water2infilt_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tmc .EQ. tmc_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_gpu are equal to tmc_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_gpu do not match tmc_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc - tmc_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_gpu:', MINVAL(tmc), MAXVAL(tmc)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_cpu:', MINVAL(tmc_cpu), MAXVAL(tmc_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(resdist .EQ. resdist_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in resdist_gpu are equal to resdist_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in resdist_gpu do not match resdist_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(resdist - resdist_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of resdist_gpu:', MINVAL(resdist), MAXVAL(resdist)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of resdist_cpu:', MINVAL(resdist_cpu), MAXVAL(resdist_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(drain_upd .EQ. drain_upd_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in drain_upd_gpu are equal to drain_upd_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in drain_upd_gpu do not match drain_upd_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(drain_upd - drain_upd_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drain_upd_gpu:', MINVAL(drain_upd), MAXVAL(drain_upd)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of drain_upd_cpu:', MINVAL(drain_upd_cpu), MAXVAL(drain_upd_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(runoff_upd .EQ. runoff_upd_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in runoff_upd_gpu are equal to runoff_upd_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in runoff_upd_gpu do not match runoff_upd_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(runoff_upd - runoff_upd_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of runoff_upd_gpu:', MINVAL(runoff_upd), MAXVAL(runoff_upd)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of runoff_upd_cpu:', MINVAL(runoff_upd_cpu), MAXVAL(runoff_upd_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(qsintveg .EQ. qsintveg_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in qsintveg_gpu are equal to qsintveg_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in qsintveg_gpu do not match qsintveg_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(qsintveg - qsintveg_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qsintveg_gpu:', MINVAL(qsintveg), MAXVAL(qsintveg)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of qsintveg_cpu:', MINVAL(qsintveg_cpu), MAXVAL(qsintveg_cpu)
    WRITE(*, *) ''
  END IF
  IF (error_flag_hydrol_tmc_update_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_tmc_update, error_flag_hydrol_tmc_update_1 is > 0 :', error_flag_hydrol_tmc_update_1
    CALL ipslerr_p(3, 'hydrol_tmc_update', 'if all resdist - see above- are zero', 'the last vegetation may have been replaced by a non biological land cover', 'This transfer has not yet been implemented in the code')
  END IF
  CONTAINS

  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_tmc_update
  !!
  !>\BRIEF        This routine updates the soil moisture profiles when the vegetation fraction have changed.
  !!
  !! DESCRIPTION  :
  !!
  !!    This routine update tmc and mc with variation of veget_max (LAND_USE or DGVM activated)
  !!
  !!
  !!
  !!
  !! RECENT CHANGE(S) : Adaptation to excluding nobio from soiltile(1)
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================

  SUBROUTINE hydrol_tmc_update_acc(error_flag_hydrol_tmc_update_1, ji, kjpindex, veget_max, soiltile, qsintveg, drain_upd, runoff_upd)
    !$ACC ROUTINE SEQ

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_tmc_update_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! domain size
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max fraction of vegetation type
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile (0-1, unitless)

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drain_upd
    !! Change in drainage due to decrease in vegtot
    !! on mc [kg/m2/dt]
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff_upd
    !! Change in runoff due to decrease in vegtot
    !! on water2infilt[kg/m2/dt]

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(INOUT) :: qsintveg
    !! Amount of water in the canopy interception

    !! 0.4 Local variables
    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: jst
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: index
    !! Indices
    LOGICAL :: soil_upd
    !! True if soiltile changed since last time step
    LOGICAL :: vegtot_upd
    !! True if vegtot changed since last time step
    REAL(KIND = r_std), DIMENSION(nstm) :: vmr
    !! Change in soiltile (within vegtot)
    REAL(KIND = r_std) :: vmr_sum
    REAL(KIND = r_std) :: delvegtot
    REAL(KIND = r_std), DIMENSION(nslm) :: mc_dilu
    !! Total loss of moisture content
    REAL(KIND = r_std) :: infil_dilu
    !! Total loss for water2infilt
    REAL(KIND = r_std), DIMENSION(nstm) :: tmc_old
    !! tmc before calculations
    REAL(KIND = r_std), DIMENSION(nstm) :: water2infilt_old
    !! water2infilt before calculations
    REAL(KIND = r_std), DIMENSION(nvm) :: qsintveg_old
    !! qsintveg before calculations
    REAL(KIND = r_std) :: test
    REAL(KIND = r_std), DIMENSION(nslm, nstm) :: mcaux
    REAL(KIND = r_std) :: minmax_value
    INTEGER(KIND = i_std) :: minmax_index
    !! serves to hold the chnage in mc when vegtot decreases


    !! 1. Update canopy interception following a land cover change
    !     If a PFT has disapperead as result from a veget_max change,
    !     the intercepted water will have been lost during the removal of the vegetation.
    !     The water previously stored on the canopy will now be added to surface water.
    !     Other adaptations of qsintveg are delt by the normal functioning of hydrol_canop
    IF (vegtot_old(ji) .GT. min_sechiba) THEN
      DO jv = 1, nvm
        IF ((veget_max(ji, jv) .LT. min_sechiba) .AND. (qsintveg(ji, jv) .GT. 0.)) THEN

          ! The PFT has been removed but there is still some water on the canopy a solution need to be
          ! found for this water. If it is a forest PFT that was removed we will just add the water to
          ! soil water column of the tall vegetation. Note that it is also possible that last forest
          ! was removed. In that case there is no longer a tall vegetation water column. In that case
          ! we need to find a different water column to add the canopy water to. Ideally that would be
          ! to water column to which the new PFT belongs. For example if the last forest became a cropland
          ! the water previously stored in the forest canopy should be added to the soil water column
          ! of the short vegetation. Because the current land cover change functionality only deals
          ! with net land cover changes we don know the exact changes. An approximation will be used.

          ! Search for a suitable soil tile index to move the canopy water into
          jst = pref_soil_veg(jv)
          IF (resdist(ji, jst) .GT. zero) THEN
            index = jst
          ELSE
            ! Note that dim=1 refers to the dimensions of the answer
            minmax_value = HUGE(0.0)
            minmax_index = 1
            DO jst = 1, nstm
              IF (resdist(ji, jst) .LT. minmax_value) THEN
                minmax_value = resdist(ji, jst)
                minmax_index = jst + 1 - 1
              END IF
            END DO
            index = minmax_index
            IF (resdist(ji, index) .LE. zero) THEN
              error_flag_hydrol_tmc_update_1 = error_flag_hydrol_tmc_update_1 + 1
            END IF
          END IF

          ! Move the canopy water into the surface water
          water2infilt(ji, index) = water2infilt(ji, index) + qsintveg(ji, jv) / (resdist(ji, index) * vegtot_old(ji))
          qsintveg(ji, jv) = zero

        END IF
      END DO
    END IF

    !! 2. We now deal with the changes of soiltile and corresponding soil moistures
    !!    Because sum(soiltile)=1 whatever vegtot, we need to distinguish two cases:
    !!    - when vegtot changes (meaning that the nobio fraction changes too),
    !!    - and when vegtot does not changes (a priori the most frequent case)

    vegtot_upd = SUM(ABS((vegtot(:) - vegtot_old(:)))) .GT. zero
    ! True if at least one land point with a vegtot change
    runoff_upd(ji) = zero
    drain_upd(ji) = zero
    IF (vegtot_upd) THEN

      ! We find here the processing specific to the chnages of nobio fraction and vegtot
      delvegtot = vegtot(ji) - vegtot_old(ji)

        DO jst = 1, nstm

          IF (delvegtot .GT. min_sechiba) THEN

          !! 2.1. If vegtot increases (nobio decreases), then the mc in each soiltile is decreased
          !!      assuming the same proportions for each soiltile, and each soil layer

          mc(ji, :, jst) = mc(ji, :, jst) * vegtot_old(ji) / vegtot(ji)
          ! vegtot cannot be zero as > vegtot_old
          water2infilt(ji, jst) = water2infilt(ji, jst) * vegtot_old(ji) / vegtot(ji)

        ELSE

            !! 2.2 If vegtot decreases (nobio increases), then the mc in each soiltile should increase,
            !!     but should not exceed mcs
            !!     For simplicity, we choose to send the corresponding water volume to drainage
            !!     We do the same for water2infilt but send the excess to surface runoff

            IF (vegtot(ji) .GT. min_sechiba) THEN
            mcaux(:, jst) = mc(ji, :, jst) * (vegtot_old(ji) - vegtot(ji)) / vegtot(ji)
            ! mcaux is the delta mc
          ELSE
            ! we just have nobio in the grid-cell
            mcaux(:, jst) = mc(ji, :, jst)
          END IF

          drain_upd(ji) = drain_upd(ji) + dz(2) * (trois * mcaux(1, jst) + mcaux(2, jst)) / huit
          DO jsl = 2, nslm - 1
            drain_upd(ji) = drain_upd(ji) + dz(jsl) * (trois * mcaux(jsl, jst) + mcaux(jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcaux(jsl, jst) + mcaux(jsl + 1, jst)) / huit
          END DO
          drain_upd(ji) = drain_upd(ji) + dz(nslm) * (trois * mcaux(nslm, jst) + mcaux(nslm - 1, jst)) / huit

            IF (vegtot(ji) .GT. min_sechiba) THEN
            runoff_upd(ji) = runoff_upd(ji) + water2infilt(ji, jst) * (vegtot_old(ji) - vegtot(ji)) / vegtot(ji)
          ELSE
            ! we just have nobio in the grid-cell
            runoff_upd(ji) = runoff_upd(ji) + water2infilt(ji, jst)
          END IF

        END IF

      END DO

    END IF

    !! 3. At the end of step 2, we are back to a case where vegtot changes are treated, so we can use soiltile
    !!    as a fraction of vegtot to process the mc transfers between soil tiles due to the changes of vegetation map

    !! 3.1 Check if soiltiles changed since last time step
    soil_upd = SUM(ABS(soiltile(:, :) - resdist(:, :))) .GT. zero

      IF (soil_upd) THEN

      !! 3.2 Define the change in soiltile
      vmr(:) = soiltile(ji, :) - resdist(ji, :)
      ! resdist is the previous values of soiltiles, previous timestep, so before new map

      ! Total area loss by the three soil tiles
      vmr_sum = SUM(vmr(:), MASK = vmr(:) .LT. zero)

      !! 3.3 Shrinking soil tiles
      !! 3.3.1 Total loss of moisture content from the shrinking soil tiles, expressed by soil layer
      mc_dilu(:) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          IF (vmr(jst) < - min_sechiba) THEN
            mc_dilu(jsl) = mc_dilu(jsl) + mc(ji, jsl, jst) * vmr(jst) / vmr_sum
          END IF
        END DO
      END DO

      !! 3.3.2 Total loss of water2inft from the shrinking soil tiles
      infil_dilu = zero
      DO jst = 1, nstm
        IF (vmr(jst) < - min_sechiba) THEN
          infil_dilu = infil_dilu + water2infilt(ji, jst) * vmr(jst) / vmr_sum
        END IF
      END DO

        !! 3.4 Each gaining soil tile gets moisture proportionally to both the total loss and its areal increase

        ! As the original mc from each soil tile are in [mcr,mcs] and we do weighted avrage, the new mc are in [mcr,mcs]
        ! The case where the soiltile is created (soiltile_old=0) works as the other cases

        ! 3.4.1 Update mc(kjpindex,nslm,nstm) !m3/m3
        DO jst = 1, nstm
        DO jsl = 1, nslm
          IF (vmr(jst) > min_sechiba) THEN
            mc(ji, jsl, jst) = (mc(ji, jsl, jst) * resdist(ji, jst) + mc_dilu(jsl) * vmr(jst)) / soiltile(ji, jst)
            ! NB : soiltile can not be zero for case vmr > zero, see slowproc_veget
          END IF
        END DO
      END DO

        ! 3.4.2 Update water2inft
        DO jst = 1, nstm
        IF (vmr(jst) > min_sechiba) THEN
          !donc soiltile>0
          water2infilt(ji, jst) = (water2infilt(ji, jst) * resdist(ji, jst) + infil_dilu * vmr(jst)) / soiltile(ji, jst)
        END IF
        !donc resdist>0
      END DO

        ! 3.4.3 Case where soiltile < min_sechiba
        DO jst = 1, nstm
        IF (soiltile(ji, jst) .LT. min_sechiba) THEN
          water2infilt(ji, jst) = zero
          mc(ji, :, jst) = zero
        END IF
      END DO

    END IF
    ! soil_upd

      !! 4. Update tmc and humtot

      DO jst = 1, nstm
      tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
      END DO
      tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
      tmc(ji, jst) = tmc(ji, jst) + water2infilt(ji, jst)
      ! WARNING tmc is increased by includes water2infilt(ji,jst)
    END DO

    humtot(ji) = zero
    DO jst = 1, nstm
      humtot(ji) = humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst)
      ! average over grid-cell (i.e. total land)
    END DO


    !! Now that the work is done, update resdist
    resdist(ji, :) = soiltile(ji, :)


  END SUBROUTINE hydrol_tmc_update_acc

    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_tmc_update
    !!
    !>\BRIEF        This routine updates the soil moisture profiles when the vegetation fraction have changed.
    !!
    !! DESCRIPTION  :
    !!
    !!    This routine update tmc and mc with variation of veget_max (LAND_USE or DGVM activated)
    !!
    !!
    !!
    !!
    !! RECENT CHANGE(S) : Adaptation to excluding nobio from soiltile(1)
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================

    SUBROUTINE hydrol_tmc_update(kjpindex, veget_max, soiltile, qsintveg, drain_upd, runoff_upd)

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! domain size
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max fraction of vegetation type
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soil tile (0-1, unitless)

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: drain_upd
    !! Change in drainage due to decrease in vegtot
    !! on mc [kg/m2/dt]
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: runoff_upd
    !! Change in runoff due to decrease in vegtot
    !! on water2infilt[kg/m2/dt]

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(INOUT) :: qsintveg
    !! Amount of water in the canopy interception

    !! 0.4 Local variables
    INTEGER(KIND = i_std) :: ji, jv, jst, jsl, index
    !! Indices
    LOGICAL :: soil_upd
    !! True if soiltile changed since last time step
    LOGICAL :: vegtot_upd
    !! True if vegtot changed since last time step
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: vmr
    !! Change in soiltile (within vegtot)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: vmr_sum
    REAL(KIND = r_std), DIMENSION(kjpindex) :: delvegtot
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mc_dilu
    !! Total loss of moisture content
    REAL(KIND = r_std), DIMENSION(kjpindex) :: infil_dilu
    !! Total loss for water2infilt
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: tmc_old
    !! tmc before calculations
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: water2infilt_old
    !! water2infilt before calculations
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: qsintveg_old
    !! qsintveg before calculations
    REAL(KIND = r_std), DIMENSION(kjpindex) :: test
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mcaux
    !! serves to hold the chnage in mc when vegtot decreases


    !! 1. Update canopy interception following a land cover change
    !     If a PFT has disapperead as result from a veget_max change,
    !     the intercepted water will have been lost during the removal of the vegetation.
    !     The water previously stored on the canopy will now be added to surface water.
    !     Other adaptations of qsintveg are delt by the normal functioning of hydrol_canop
    DO ji = 1, kjpindex
      IF (vegtot_old(ji) .GT. min_sechiba) THEN
        DO jv = 1, nvm
          IF ((veget_max(ji, jv) .LT. min_sechiba) .AND. (qsintveg(ji, jv) .GT. 0.)) THEN

            ! The PFT has been removed but there is still some water on the canopy a solution need to be
            ! found for this water. If it is a forest PFT that was removed we will just add the water to
            ! soil water column of the tall vegetation. Note that it is also possible that last forest
            ! was removed. In that case there is no longer a tall vegetation water column. In that case
            ! we need to find a different water column to add the canopy water to. Ideally that would be
            ! to water column to which the new PFT belongs. For example if the last forest became a cropland
            ! the water previously stored in the forest canopy should be added to the soil water column
            ! of the short vegetation. Because the current land cover change functionality only deals
            ! with net land cover changes we don know the exact changes. An approximation will be used.

            ! Search for a suitable soil tile index to move the canopy water into
            jst = pref_soil_veg(jv)
            IF (resdist(ji, jst) .GT. zero) THEN
              index = jst
            ELSE
              ! Note that dim=1 refers to the dimensions of the answer
              index = MAXLOC(resdist(ji, :), DIM = 1)
              IF (resdist(ji, index) .LE. zero) THEN
                WRITE(numout, *) 'ipts, index, resdist, ', ji, index, resdist(ji, :)
                CALL ipslerr_p(3, 'hydrol_tmc_update', 'if all resdist - see above- are zero', 'the last vegetation may have been replaced by a non biological land cover', 'This transfer has not yet been implemented in the code')
              END IF
            END IF

            ! Move the canopy water into the surface water
            water2infilt(ji, index) = water2infilt(ji, index) + qsintveg(ji, jv) / (resdist(ji, index) * vegtot_old(ji))
            qsintveg(ji, jv) = zero

          END IF
        END DO
      END IF
    END DO

    !! 2. We now deal with the changes of soiltile and corresponding soil moistures
    !!    Because sum(soiltile)=1 whatever vegtot, we need to distinguish two cases:
    !!    - when vegtot changes (meaning that the nobio fraction changes too),
    !!    - and when vegtot does not changes (a priori the most frequent case)

    vegtot_upd = SUM(ABS((vegtot(:) - vegtot_old(:)))) .GT. zero
    ! True if at least one land point with a vegtot change
    runoff_upd(:) = zero
    drain_upd(:) = zero
    IF (vegtot_upd) THEN

      ! We find here the processing specific to the chnages of nobio fraction and vegtot
      delvegtot(:) = vegtot(:) - vegtot_old(:)

        DO jst = 1, nstm
        DO ji = 1, kjpindex

            IF (delvegtot(ji) .GT. min_sechiba) THEN

            !! 2.1. If vegtot increases (nobio decreases), then the mc in each soiltile is decreased
            !!      assuming the same proportions for each soiltile, and each soil layer

            mc(ji, :, jst) = mc(ji, :, jst) * vegtot_old(ji) / vegtot(ji)
            ! vegtot cannot be zero as > vegtot_old
            water2infilt(ji, jst) = water2infilt(ji, jst) * vegtot_old(ji) / vegtot(ji)

          ELSE

              !! 2.2 If vegtot decreases (nobio increases), then the mc in each soiltile should increase,
              !!     but should not exceed mcs
              !!     For simplicity, we choose to send the corresponding water volume to drainage
              !!     We do the same for water2infilt but send the excess to surface runoff

              IF (vegtot(ji) .GT. min_sechiba) THEN
              mcaux(ji, :, jst) = mc(ji, :, jst) * (vegtot_old(ji) - vegtot(ji)) / vegtot(ji)
              ! mcaux is the delta mc
            ELSE
              ! we just have nobio in the grid-cell
              mcaux(ji, :, jst) = mc(ji, :, jst)
            END IF

            drain_upd(ji) = drain_upd(ji) + dz(2) * (trois * mcaux(ji, 1, jst) + mcaux(ji, 2, jst)) / huit
            DO jsl = 2, nslm - 1
              drain_upd(ji) = drain_upd(ji) + dz(jsl) * (trois * mcaux(ji, jsl, jst) + mcaux(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mcaux(ji, jsl, jst) + mcaux(ji, jsl + 1, jst)) / huit
            END DO
            drain_upd(ji) = drain_upd(ji) + dz(nslm) * (trois * mcaux(ji, nslm, jst) + mcaux(ji, nslm - 1, jst)) / huit

              IF (vegtot(ji) .GT. min_sechiba) THEN
              runoff_upd(ji) = runoff_upd(ji) + water2infilt(ji, jst) * (vegtot_old(ji) - vegtot(ji)) / vegtot(ji)
            ELSE
              ! we just have nobio in the grid-cell
              runoff_upd(ji) = runoff_upd(ji) + water2infilt(ji, jst)
            END IF

          END IF

        END DO
      END DO

    END IF

    !! 3. At the end of step 2, we are back to a case where vegtot changes are treated, so we can use soiltile
    !!    as a fraction of vegtot to process the mc transfers between soil tiles due to the changes of vegetation map

    !! 3.1 Check if soiltiles changed since last time step
    soil_upd = SUM(ABS(soiltile(:, :) - resdist(:, :))) .GT. zero
    IF (printlev >= 3) WRITE(numout, *) 'soil_upd ', soil_upd

      IF (soil_upd) THEN

      !! 3.2 Define the change in soiltile
      vmr(:, :) = soiltile(:, :) - resdist(:, :)
      ! resdist is the previous values of soiltiles, previous timestep, so before new map

        ! Total area loss by the three soil tiles
        DO ji = 1, kjpindex
        vmr_sum(ji) = SUM(vmr(ji, :), MASK = vmr(ji, :) .LT. zero)
      END DO

      !! 3.3 Shrinking soil tiles
      !! 3.3.1 Total loss of moisture content from the shrinking soil tiles, expressed by soil layer
      mc_dilu(:, :) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF (vmr(ji, jst) < - min_sechiba) THEN
              mc_dilu(ji, jsl) = mc_dilu(ji, jsl) + mc(ji, jsl, jst) * vmr(ji, jst) / vmr_sum(ji)
            END IF
          END DO
        END DO
      END DO

      !! 3.3.2 Total loss of water2inft from the shrinking soil tiles
      infil_dilu(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (vmr(ji, jst) < - min_sechiba) THEN
            infil_dilu(ji) = infil_dilu(ji) + water2infilt(ji, jst) * vmr(ji, jst) / vmr_sum(ji)
          END IF
        END DO
      END DO

        !! 3.4 Each gaining soil tile gets moisture proportionally to both the total loss and its areal increase

        ! As the original mc from each soil tile are in [mcr,mcs] and we do weighted avrage, the new mc are in [mcr,mcs]
        ! The case where the soiltile is created (soiltile_old=0) works as the other cases

        ! 3.4.1 Update mc(kjpindex,nslm,nstm) !m3/m3
        DO jst = 1, nstm
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF (vmr(ji, jst) > min_sechiba) THEN
              mc(ji, jsl, jst) = (mc(ji, jsl, jst) * resdist(ji, jst) + mc_dilu(ji, jsl) * vmr(ji, jst)) / soiltile(ji, jst)
              ! NB : soiltile can not be zero for case vmr > zero, see slowproc_veget
            END IF
          END DO
        END DO
      END DO

        ! 3.4.2 Update water2inft
        DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (vmr(ji, jst) > min_sechiba) THEN
            !donc soiltile>0
            water2infilt(ji, jst) = (water2infilt(ji, jst) * resdist(ji, jst) + infil_dilu(ji) * vmr(ji, jst)) / soiltile(ji, jst)
          END IF
          !donc resdist>0
        END DO
      END DO

        ! 3.4.3 Case where soiltile < min_sechiba
        DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (soiltile(ji, jst) .LT. min_sechiba) THEN
            water2infilt(ji, jst) = zero
            mc(ji, :, jst) = zero
          END IF
        END DO
      END DO

    END IF
    ! soil_upd

      !! 4. Update tmc and humtot

      DO jst = 1, nstm
      DO ji = 1, kjpindex
        tmc(ji, jst) = dz(2) * (trois * mc(ji, 1, jst) + mc(ji, 2, jst)) / huit
        DO jsl = 2, nslm - 1
          tmc(ji, jst) = tmc(ji, jst) + dz(jsl) * (trois * mc(ji, jsl, jst) + mc(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc(ji, jsl, jst) + mc(ji, jsl + 1, jst)) / huit
        END DO
        tmc(ji, jst) = tmc(ji, jst) + dz(nslm) * (trois * mc(ji, nslm, jst) + mc(ji, nslm - 1, jst)) / huit
        tmc(ji, jst) = tmc(ji, jst) + water2infilt(ji, jst)
        ! WARNING tmc is increased by includes water2infilt(ji,jst)
      END DO
    END DO

    humtot(:) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        humtot(ji) = humtot(ji) + vegtot(ji) * soiltile(ji, jst) * tmc(ji, jst)
        ! average over grid-cell (i.e. total land)
      END DO
    END DO


    !! Now that the work is done, update resdist
    resdist(:, :) = soiltile(:, :)

    IF (printlev >= 3) WRITE(numout, *) ' hydrol_tmc_update done '

  END SUBROUTINE hydrol_tmc_update
  SUBROUTINE read_dummy(veget_max, soiltile, qsintveg, ji, error_flag_hydrol_tmc_update_1)
    INTEGER(KIND = i_std) :: error_flag_hydrol_tmc_update_1
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: qsintveg
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    CALL random_number(veget_max)
    CALL random_number(soiltile)
    CALL random_number(qsintveg)
    ji = 2
    error_flag_hydrol_tmc_update_1 = 2
  END SUBROUTINE read_dummy
END PROGRAM main
