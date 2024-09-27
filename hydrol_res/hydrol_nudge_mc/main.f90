
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: jst
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mc_loc, mc_loc_cpu
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(jst, mc_loc, ji)
  CALL hydrol_nudge_mc(kjpindex, jst, mc_loc)
  tmc_aux_cpu = tmc_aux
  mc_loc_cpu = mc_loc
  CALL initialization
  CALL read_dummy(jst, mc_loc, ji)
  !$ACC ENTER DATA COPYIN(mc_loc)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_nudge_mc_acc(ji, kjpindex, jst, mc_loc)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(tmc_aux, mc_loc)
  !$ACC EXIT DATA DELETE(mc_loc)
  IF (ALL(tmc_aux .EQ. tmc_aux_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tmc_aux_gpu are equal to tmc_aux_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tmc_aux_gpu do not match tmc_aux_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tmc_aux - tmc_aux_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_aux_gpu:', MINVAL(tmc_aux), MAXVAL(tmc_aux)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tmc_aux_cpu:', MINVAL(tmc_aux_cpu), MAXVAL(tmc_aux_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(mc_loc .EQ. mc_loc_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mc_loc_gpu are equal to mc_loc_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mc_loc_gpu do not match mc_loc_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mc_loc - mc_loc_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_loc_gpu:', MINVAL(mc_loc), MAXVAL(mc_loc)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mc_loc_cpu:', MINVAL(mc_loc_cpu), MAXVAL(mc_loc_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS

  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_nudge_mc
  !!
  !>\BRIEF         Applay nuding for soil moisture
  !!
  !! DESCRIPTION  : Applay nudging for soil moisture. The nuding values were previously read and interpolated using
  !!                the subroutine hydrol_nudge_mc_read
  !!                This subroutine is called from a loop over all soil tiles.
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! \n
  !_ ================================================================================================================================
  SUBROUTINE hydrol_nudge_mc_acc(ji, kjpindex, jst, mc_loc)
    !$ACC ROUTINE SEQ

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: jst
    !! Index for current soil tile

    !! 0.2 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(INOUT) :: mc_loc
    !! Soil moisture

    !! 0.2 Locals variables
    REAL(KIND = r_std), DIMENSION(nslm, nstm) :: mc_aux
    !! Temorary variable for calculation of nudgincsm
    INTEGER(KIND = i_std) :: jsl
    !! loop index


    !! 1.5 Applay nudging of soil moisture using alpha_nudge_mc at each model sechiba time step.
    !!     alpha_mc_nudge calculated using the parameter for relaxation time NUDGE_TAU_MC set in module constantes.
    !!     alpha_nudge_mc is between 0-1
    !!     If alpha_nudge_mc=1, the new mc will be replaced by the one read from file
    mc_loc(ji, :, jst) = (1 - alpha_nudge_mc) * mc_loc(ji, :, jst) + alpha_nudge_mc * mc_read_current(ji, :, jst)


    !! 1.6 Calculate diagnostic for nudging increment of water in soil moisture
    !!     Here calculate tmc_aux for the current soil tile. Later in hydrol_nudge_mc_diag, this will be used to calculate nudgincsm
    mc_aux(:, jst) = alpha_nudge_mc * (mc_read_current(ji, :, jst) - mc_loc(ji, :, jst))
    tmc_aux(ji, jst) = dz(2) * (trois * mc_aux(1, jst) + mc_aux(2, jst)) / huit
    DO jsl = 2, nslm - 1
      tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(jsl) * (trois * mc_aux(jsl, jst) + mc_aux(jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc_aux(jsl, jst) + mc_aux(jsl + 1, jst)) / huit
    END DO
    tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(nslm) * (trois * mc_aux(nslm, jst) + mc_aux(nslm - 1, jst)) / huit


  END SUBROUTINE hydrol_nudge_mc_acc

    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_nudge_mc
    !!
    !>\BRIEF         Applay nuding for soil moisture
    !!
    !! DESCRIPTION  : Applay nudging for soil moisture. The nuding values were previously read and interpolated using
    !!                the subroutine hydrol_nudge_mc_read
    !!                This subroutine is called from a loop over all soil tiles.
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! \n
    !_ ================================================================================================================================
    SUBROUTINE hydrol_nudge_mc(kjpindex, jst, mc_loc)

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: jst
    !! Index for current soil tile

    !! 0.2 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm), INTENT(INOUT) :: mc_loc
    !! Soil moisture

    !! 0.2 Locals variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mc_aux
    !! Temorary variable for calculation of nudgincsm
    INTEGER(KIND = i_std) :: ji, jsl
    !! loop index


    !! 1.5 Applay nudging of soil moisture using alpha_nudge_mc at each model sechiba time step.
    !!     alpha_mc_nudge calculated using the parameter for relaxation time NUDGE_TAU_MC set in module constantes.
    !!     alpha_nudge_mc is between 0-1
    !!     If alpha_nudge_mc=1, the new mc will be replaced by the one read from file
    mc_loc(:, :, jst) = (1 - alpha_nudge_mc) * mc_loc(:, :, jst) + alpha_nudge_mc * mc_read_current(:, :, jst)


    !! 1.6 Calculate diagnostic for nudging increment of water in soil moisture
    !!     Here calculate tmc_aux for the current soil tile. Later in hydrol_nudge_mc_diag, this will be used to calculate nudgincsm
    mc_aux(:, :, jst) = alpha_nudge_mc * (mc_read_current(:, :, jst) - mc_loc(:, :, jst))
    DO ji = 1, kjpindex
      tmc_aux(ji, jst) = dz(2) * (trois * mc_aux(ji, 1, jst) + mc_aux(ji, 2, jst)) / huit
      DO jsl = 2, nslm - 1
        tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(jsl) * (trois * mc_aux(ji, jsl, jst) + mc_aux(ji, jsl - 1, jst)) / huit + dz(jsl + 1) * (trois * mc_aux(ji, jsl, jst) + mc_aux(ji, jsl + 1, jst)) / huit
      END DO
      tmc_aux(ji, jst) = tmc_aux(ji, jst) + dz(nslm) * (trois * mc_aux(ji, nslm, jst) + mc_aux(ji, nslm - 1, jst)) / huit
    END DO


  END SUBROUTINE hydrol_nudge_mc
  SUBROUTINE read_dummy(jst, mc_loc, ji)
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm, nstm) :: mc_loc
    INTEGER(KIND = i_std) :: jst
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    jst = 2
    CALL random_number(mc_loc)
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
