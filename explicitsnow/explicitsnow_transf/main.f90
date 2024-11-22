
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz, snowdz_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz_old
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowgrain, snowgrain_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowheat, snowheat_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowrho, snowrho_cpu
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declaration_initialization
  CALL read_dummy(snowdz, snowdz_old, snowgrain, snowheat, snowrho)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  CALL explicitsnow_transf(kjpindex, snowdz_old, snowdz, snowrho, snowheat, snowgrain)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  snowdz_cpu = snowdz
  snowgrain_cpu = snowgrain
  snowheat_cpu = snowheat
  snowrho_cpu = snowrho
  CALL declaration_initialization
  CALL read_dummy(snowdz, snowdz_old, snowgrain, snowheat, snowrho)
  !$ACC ENTER DATA COPYIN(snowdz_old, snowdz, snowrho, snowheat, snowgrain)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL explicitsnow_transf_acc(ji, kjpindex, snowdz_old, snowdz, snowrho, snowheat, snowgrain)
  END DO
  !$ACC END PARALLEL
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  !$ACC UPDATE SELF(snowdz, snowgrain, snowheat, snowrho)
  !$ACC EXIT DATA DELETE(snowdz_old, snowdz, snowrho, snowheat, snowgrain)
  IF (ALL(snowdz .EQ. snowdz_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snowdz_gpu are equal to snowdz_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snowdz_gpu do not match snowdz_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snowdz - snowdz_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowdz_gpu:', MINVAL(snowdz), MAXVAL(snowdz)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowdz_cpu:', MINVAL(snowdz_cpu), MAXVAL(snowdz_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(snowgrain .EQ. snowgrain_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snowgrain_gpu are equal to snowgrain_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snowgrain_gpu do not match snowgrain_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snowgrain - snowgrain_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowgrain_gpu:', MINVAL(snowgrain), MAXVAL(snowgrain)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowgrain_cpu:', MINVAL(snowgrain_cpu), MAXVAL(snowgrain_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(snowheat .EQ. snowheat_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snowheat_gpu are equal to snowheat_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snowheat_gpu do not match snowheat_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snowheat - snowheat_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowheat_gpu:', MINVAL(snowheat), MAXVAL(snowheat)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowheat_cpu:', MINVAL(snowheat_cpu), MAXVAL(snowheat_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(snowrho .EQ. snowrho_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snowrho_gpu are equal to snowrho_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snowrho_gpu do not match snowrho_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snowrho - snowrho_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowrho_gpu:', MINVAL(snowrho), MAXVAL(snowrho)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snowrho_cpu:', MINVAL(snowrho_cpu), MAXVAL(snowrho_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS

  !!
  !================================================================================================================================
  !! SUBROUTINE   : explicitsnow_transf
  !!
  !>\BRIEF        Computing snow mass and heat redistribution due to grid thickness configuration resetting
  !!
  !! DESCRIPTION  : Snow mass and heat redistibution due to grid thickness
  !!                configuration resetting. Total mass and heat content
  !!                of the overall snowpack unchanged/conserved within this routine.
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S): None
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !================================================================================================================================

  SUBROUTINE explicitsnow_transf_acc(ji, kjpindex, snowdz_old, snowdz, snowrho, snowheat, snowgrain)
    !$ACC ROUTINE SEQ

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(IN) :: snowdz_old
    !! Snow depth at the previous time step

    !! 0.2 Output variables

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowrho
    !! Snow density (Kg/m^3)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowgrain
    !! Snow grain size (m)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowdz
    !! Snow depth (m)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowheat
    !! Snow heat content/enthalpy (J/m2)

    !! 0.4 Local varibles
    REAL(KIND = r_std), DIMENSION(0 : nsnow) :: zsnowzo
    REAL(KIND = r_std), DIMENSION(0 : nsnow) :: zsnowzn
    REAL(KIND = r_std), DIMENSION(nsnow) :: zsnowddz
    REAL(KIND = r_std), DIMENSION(nsnow) :: zdelta
    REAL(KIND = r_std), DIMENSION(nsnow) :: zsnowgrainn
    REAL(KIND = r_std), DIMENSION(nsnow) :: zsnowheatn
    REAL(KIND = r_std), DIMENSION(nsnow) :: zsnowrhon
    REAL(KIND = r_std) :: zsnowmix_delta
    REAL(KIND = r_std) :: zsumgrain
    REAL(KIND = r_std) :: zsumswe
    REAL(KIND = r_std) :: zsumheat
    INTEGER(KIND = i_std), DIMENSION(nsnow, 2) :: locflag
    REAL(KIND = r_std) :: psnow
    INTEGER(KIND = i_std) :: jjj
    INTEGER(KIND = i_std) :: jj

    ! Initialization
    zsumheat = 0.0
    zsumswe = 0.0
    zsumgrain = 0.0
    zsnowmix_delta = 0.0
    locflag(:, :) = 0

    psnow = SUM(snowdz(ji, :))

      IF (psnow .GE. xsnowcritd .AND. snowdz_old(ji, 1) .NE. 0 .AND. snowdz_old(ji, 2) .NE. 0 .AND. snowdz_old(ji, 3) .NE. 0) THEN
      zsnowzo(0) = 0.
      zsnowzn(0) = 0.
      zsnowzo(1) = snowdz_old(ji, 1)
      zsnowzn(1) = snowdz(ji, 1)

        DO jj = 2, nsnow
        zsnowzo(jj) = zsnowzo(jj - 1) + snowdz_old(ji, jj)
        zsnowzn(jj) = zsnowzn(jj - 1) + snowdz(ji, jj)
      END DO

        DO jj = 1, nsnow
        IF (jj .EQ. 1) THEN
          locflag(jj, 1) = 1
        ELSE
          DO jjj = nsnow, 1, - 1
            !upper bound of the snow layer
              IF (zsnowzn(jj - 1) .LE. zsnowzo(jjj)) THEN
              locflag(jj, 1) = jjj
            END IF
          END DO
        END IF

          IF (jj .EQ. nsnow) THEN
          locflag(jj, 2) = nsnow
        ELSE
          DO jjj = nsnow, 1, - 1
            !lower bound of the snow layer
              IF (zsnowzn(jj) .LE. zsnowzo(jjj)) THEN
              locflag(jj, 2) = jjj
            END IF
          END DO
        END IF

          !to interpolate
          ! when heavy snow occurred
          IF (locflag(jj, 1) .EQ. locflag(jj, 2)) THEN
          zsnowrhon(jj) = snowrho(ji, locflag(jj, 1))

          zsnowheatn(jj) = snowheat(ji, locflag(jj, 1)) * snowdz(ji, jj) / snowdz_old(ji, locflag(jj, 1))

          zsnowgrainn(jj) = snowgrain(ji, locflag(jj, 1))
        ELSE
          !snow density
          zsnowrhon(jj) = snowrho(ji, locflag(jj, 1)) * (zsnowzo(locflag(jj, 1)) - zsnowzn(jj - 1)) + snowrho(ji, locflag(jj, 2)) &
&* (zsnowzn(jj) - zsnowzo(locflag(jj, 2) - 1))

            DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
            zsnowrhon(jj) = zsnowrhon(jj) + (jjj - locflag(jj, 1)) * snowrho(ji, jjj) * snowdz_old(ji, jjj)
          END DO

          zsnowrhon(jj) = zsnowrhon(jj) / snowdz(ji, jj)

            !snow heat
            IF (snowdz_old(ji, locflag(jj, 1)) .GT. 0.0) THEN
            zsnowheatn(jj) = snowheat(ji, locflag(jj, 1)) * (zsnowzo(locflag(jj, 1)) - zsnowzn(jj - 1)) / snowdz_old(ji, &
&locflag(jj, 1)) + snowheat(ji, locflag(jj, 2)) * (zsnowzn(jj) - zsnowzo(locflag(jj, 2) - 1)) / snowdz_old(ji, locflag(jj, 2))
          ELSE
            zsnowheatn(jj) = snowheat(ji, locflag(jj, 2)) / snowdz_old(ji, locflag(jj, 2)) * (zsnowzn(locflag(jj, 1)) - &
&zsnowzo(locflag(jj, 1)))
          END IF

            DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
            zsnowheatn(jj) = zsnowheatn(jj) + (jjj - locflag(jj, 1)) * snowheat(ji, jjj)
          END DO

          !snow grain
          zsnowgrainn(jj) = snowgrain(ji, locflag(jj, 1)) * (zsnowzo(locflag(jj, 1)) - zsnowzn(jj - 1)) + snowgrain(ji, &
&locflag(jj, 2)) * (zsnowzn(jj) - zsnowzo(locflag(jj, 2) - 1))

            DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
            zsnowgrainn(jj) = zsnowgrainn(jj) + (jjj - locflag(jj, 1)) * snowgrain(ji, jjj) * snowdz_old(ji, jjj)
          END DO
          zsnowgrainn(jj) = zsnowgrainn(jj) / snowdz(ji, jj)
        END IF
      END DO
      snowrho(ji, :) = zsnowrhon(:)
      snowheat(ji, :) = zsnowheatn(:)
      snowgrain(ji, :) = zsnowgrainn(:)
    END IF

      ! Vanishing or very thin snowpack check:
      ! -----------------------------------------
      !
      ! NOTE: ONLY for very shallow snowpacks, mix properties (homogeneous):
      ! this avoids problems related to heat and mass exchange for
      ! thin layers during heavy snowfall or signifigant melt: one
      ! new/old layer can exceed the thickness of several old/new layers.
      ! Therefore, mix (conservative):
      !
      ! modified by Tao Wang
      IF (psnow > 0 .AND. (psnow < xsnowcritd .OR. snowdz_old(ji, 1) .EQ. 0 .OR. snowdz_old(ji, 2) .EQ. 0 .OR. snowdz_old(ji, 3) &
&.EQ. 0)) THEN
      zsumheat = SUM(snowheat(ji, :))
      zsumswe = SUM(snowrho(ji, :) * snowdz_old(ji, :))
      zsumgrain = SUM(snowgrain(ji, :) * snowdz_old(ji, :))
      zsnowmix_delta = 1.0
      DO jj = 1, nsnow
        zsnowheatn(jj) = zsnowmix_delta * (zsumheat / nsnow)
        snowdz(ji, jj) = zsnowmix_delta * (psnow / nsnow)
        zsnowrhon(jj) = zsnowmix_delta * (zsumswe / psnow)
        zsnowgrainn(jj) = zsnowmix_delta * (zsumgrain / psnow)
      END DO
      ! Update mass (density and thickness), heat and grain size:
      ! ------------------------------------------------------------
      snowrho(ji, :) = zsnowrhon(:)
      snowheat(ji, :) = zsnowheatn(:)
      snowgrain(ji, :) = zsnowgrainn(:)
    END IF

  END SUBROUTINE explicitsnow_transf_acc

    !!
    
!& !================================================================================================================================
    !! SUBROUTINE   : explicitsnow_transf
    !!
    !>\BRIEF        Computing snow mass and heat redistribution due to grid thickness configuration resetting
    !!
    !! DESCRIPTION  : Snow mass and heat redistibution due to grid thickness
    !!                configuration resetting. Total mass and heat content
    !!                of the overall snowpack unchanged/conserved within this routine.
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S): None
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    
!& !================================================================================================================================

    SUBROUTINE explicitsnow_transf(kjpindex, snowdz_old, snowdz, snowrho, snowheat, snowgrain)

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(IN) :: snowdz_old
    !! Snow depth at the previous time step

    !! 0.2 Output variables

    !! 0.3 Modified variables
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowrho
    !! Snow density (Kg/m^3)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowgrain
    !! Snow grain size (m)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowdz
    !! Snow depth (m)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowheat
    !! Snow heat content/enthalpy (J/m2)

    !! 0.4 Local varibles
    REAL(KIND = r_std), DIMENSION(kjpindex, 0 : nsnow) :: zsnowzo
    REAL(KIND = r_std), DIMENSION(kjpindex, 0 : nsnow) :: zsnowzn
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: zsnowddz
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: zdelta
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: zsnowgrainn
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: zsnowheatn
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: zsnowrhon
    REAL(KIND = r_std), DIMENSION(kjpindex) :: zsnowmix_delta
    REAL(KIND = r_std), DIMENSION(kjpindex) :: zsumgrain
    REAL(KIND = r_std), DIMENSION(kjpindex) :: zsumswe
    REAL(KIND = r_std), DIMENSION(kjpindex) :: zsumheat
    INTEGER(KIND = i_std), DIMENSION(nsnow, 2) :: locflag
    REAL(KIND = r_std) :: psnow
    INTEGER(KIND = i_std) :: jjj
    INTEGER(KIND = i_std) :: jj
    INTEGER(KIND = i_std) :: ji

    ! Initialization
    zsumheat(:) = 0.0
    zsumswe(:) = 0.0
    zsumgrain(:) = 0.0
    zsnowmix_delta(:) = 0.0
    locflag(:, :) = 0

      DO ji = 1, kjpindex
      psnow = SUM(snowdz(ji, :))

        IF (psnow .GE. xsnowcritd .AND. snowdz_old(ji, 1) .NE. 0 .AND. snowdz_old(ji, 2) .NE. 0 .AND. snowdz_old(ji, 3) .NE. 0) THEN
        zsnowzo(ji, 0) = 0.
        zsnowzn(ji, 0) = 0.
        zsnowzo(ji, 1) = snowdz_old(ji, 1)
        zsnowzn(ji, 1) = snowdz(ji, 1)

          DO jj = 2, nsnow
          zsnowzo(ji, jj) = zsnowzo(ji, jj - 1) + snowdz_old(ji, jj)
          zsnowzn(ji, jj) = zsnowzn(ji, jj - 1) + snowdz(ji, jj)
        END DO

          DO jj = 1, nsnow
          IF (jj .EQ. 1) THEN
            locflag(jj, 1) = 1
          ELSE
            DO jjj = nsnow, 1, - 1
              !upper bound of the snow layer
                IF (zsnowzn(ji, jj - 1) .LE. zsnowzo(ji, jjj)) THEN
                locflag(jj, 1) = jjj
              END IF
            END DO
          END IF

            IF (jj .EQ. nsnow) THEN
            locflag(jj, 2) = nsnow
          ELSE
            DO jjj = nsnow, 1, - 1
              !lower bound of the snow layer
                IF (zsnowzn(ji, jj) .LE. zsnowzo(ji, jjj)) THEN
                locflag(jj, 2) = jjj
              END IF
            END DO
          END IF

            !to interpolate
            ! when heavy snow occurred
            IF (locflag(jj, 1) .EQ. locflag(jj, 2)) THEN
            zsnowrhon(ji, jj) = snowrho(ji, locflag(jj, 1))

            zsnowheatn(ji, jj) = snowheat(ji, locflag(jj, 1)) * snowdz(ji, jj) / snowdz_old(ji, locflag(jj, 1))

            zsnowgrainn(ji, jj) = snowgrain(ji, locflag(jj, 1))
          ELSE
            !snow density
            zsnowrhon(ji, jj) = snowrho(ji, locflag(jj, 1)) * (zsnowzo(ji, locflag(jj, 1)) - zsnowzn(ji, jj - 1)) + snowrho(ji, &
&locflag(jj, 2)) * (zsnowzn(ji, jj) - zsnowzo(ji, locflag(jj, 2) - 1))

              DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
              zsnowrhon(ji, jj) = zsnowrhon(ji, jj) + (jjj - locflag(jj, 1)) * snowrho(ji, jjj) * snowdz_old(ji, jjj)
            END DO

            zsnowrhon(ji, jj) = zsnowrhon(ji, jj) / snowdz(ji, jj)

              !snow heat
              IF (snowdz_old(ji, locflag(jj, 1)) .GT. 0.0) THEN
              zsnowheatn(ji, jj) = snowheat(ji, locflag(jj, 1)) * (zsnowzo(ji, locflag(jj, 1)) - zsnowzn(ji, jj - 1)) / &
&snowdz_old(ji, locflag(jj, 1)) + snowheat(ji, locflag(jj, 2)) * (zsnowzn(ji, jj) - zsnowzo(ji, locflag(jj, 2) - 1)) / &
&snowdz_old(ji, locflag(jj, 2))
            ELSE
              zsnowheatn(ji, jj) = snowheat(ji, locflag(jj, 2)) / snowdz_old(ji, locflag(jj, 2)) * (zsnowzn(ji, locflag(jj, 1)) - &
&zsnowzo(ji, locflag(jj, 1)))
            END IF

              DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
              zsnowheatn(ji, jj) = zsnowheatn(ji, jj) + (jjj - locflag(jj, 1)) * snowheat(ji, jjj)
            END DO

            !snow grain
            zsnowgrainn(ji, jj) = snowgrain(ji, locflag(jj, 1)) * (zsnowzo(ji, locflag(jj, 1)) - zsnowzn(ji, jj - 1)) + &
&snowgrain(ji, locflag(jj, 2)) * (zsnowzn(ji, jj) - zsnowzo(ji, locflag(jj, 2) - 1))

              DO jjj = locflag(jj, 1), locflag(jj, 2) - 1
              zsnowgrainn(ji, jj) = zsnowgrainn(ji, jj) + (jjj - locflag(jj, 1)) * snowgrain(ji, jjj) * snowdz_old(ji, jjj)
            END DO
            zsnowgrainn(ji, jj) = zsnowgrainn(ji, jj) / snowdz(ji, jj)
          END IF
        END DO
        snowrho(ji, :) = zsnowrhon(ji, :)
        snowheat(ji, :) = zsnowheatn(ji, :)
        snowgrain(ji, :) = zsnowgrainn(ji, :)
      END IF

        ! Vanishing or very thin snowpack check:
        ! -----------------------------------------
        !
        ! NOTE: ONLY for very shallow snowpacks, mix properties (homogeneous):
        ! this avoids problems related to heat and mass exchange for
        ! thin layers during heavy snowfall or signifigant melt: one
        ! new/old layer can exceed the thickness of several old/new layers.
        ! Therefore, mix (conservative):
        !
        ! modified by Tao Wang
        IF (psnow > 0 .AND. (psnow < xsnowcritd .OR. snowdz_old(ji, 1) .EQ. 0 .OR. snowdz_old(ji, 2) .EQ. 0 .OR. snowdz_old(ji, 3) &
&.EQ. 0)) THEN
        zsumheat(ji) = SUM(snowheat(ji, :))
        zsumswe(ji) = SUM(snowrho(ji, :) * snowdz_old(ji, :))
        zsumgrain(ji) = SUM(snowgrain(ji, :) * snowdz_old(ji, :))
        zsnowmix_delta(ji) = 1.0
        DO jj = 1, nsnow
          zsnowheatn(ji, jj) = zsnowmix_delta(ji) * (zsumheat(ji) / nsnow)
          snowdz(ji, jj) = zsnowmix_delta(ji) * (psnow / nsnow)
          zsnowrhon(ji, jj) = zsnowmix_delta(ji) * (zsumswe(ji) / psnow)
          zsnowgrainn(ji, jj) = zsnowmix_delta(ji) * (zsumgrain(ji) / psnow)
        END DO
        ! Update mass (density and thickness), heat and grain size:
        ! ------------------------------------------------------------
        snowrho(ji, :) = zsnowrhon(ji, :)
        snowheat(ji, :) = zsnowheatn(ji, :)
        snowgrain(ji, :) = zsnowgrainn(ji, :)
      END IF
    END DO

  END SUBROUTINE explicitsnow_transf
  SUBROUTINE read_dummy(snowdz, snowdz_old, snowgrain, snowheat, snowrho)
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowrho
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowheat
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowgrain
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz_old
    REAL(KIND = r_std), DIMENSION(kjpindex, nsnow) :: snowdz
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/explicitsnow_transf/dummy.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- inside the read dummy routine for explicitsnow_transf ---'
    READ(1363, IOSTAT = ier) snowdz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowdz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snowdz_old
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowdz_old. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snowgrain
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowgrain. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snowheat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowheat. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snowrho
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snowrho. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
  END SUBROUTINE read_dummy
END PROGRAM main
