
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
  REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
  REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_snow
  REAL(KIND = r_std), DIMENSION(kjpindex) :: snow
  REAL(KIND = r_std), DIMENSION(kjpindex) :: snow_age, snow_age_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: snow_nobio_age, snow_nobio_age_cpu
  REAL(KIND = r_std), DIMENSION(kjpindex) :: temp_sol_new
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declaration_initialization
  CALL read_dummy(frac_snow_nobio, precip_rain, precip_snow, snow, temp_sol_new)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  CALL explicitsnow_age(kjpindex, snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, snow_nobio_age)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  snow_age_cpu = snow_age
  snow_nobio_age_cpu = snow_nobio_age
  CALL declaration_initialization
  CALL read_dummy(frac_snow_nobio, precip_rain, precip_snow, snow, temp_sol_new)
  !$ACC ENTER DATA COPYIN(snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, snow_nobio_age)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL explicitsnow_age_acc(ji, kjpindex, snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, snow_nobio_age)
  END DO
  !$ACC END PARALLEL
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  !$ACC UPDATE SELF(snow_age, snow_nobio_age)
  !$ACC EXIT DATA DELETE(snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, snow_nobio_age)
  IF (ALL(snow_age .EQ. snow_age_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snow_age_gpu are equal to snow_age_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snow_age_gpu do not match snow_age_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snow_age - snow_age_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snow_age_gpu:', MINVAL(snow_age), MAXVAL(snow_age)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snow_age_cpu:', MINVAL(snow_age_cpu), MAXVAL(snow_age_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(snow_nobio_age .EQ. snow_nobio_age_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in snow_nobio_age_gpu are equal to snow_nobio_age_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in snow_nobio_age_gpu do not match snow_nobio_age_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(snow_nobio_age - snow_nobio_age_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snow_nobio_age_gpu:', MINVAL(snow_nobio_age), MAXVAL(snow_nobio_age)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of snow_nobio_age_cpu:', MINVAL(snow_nobio_age_cpu), MAXVAL(snow_nobio_age_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS


  !! 
!& ================================================================================================================================
  !! SUBROUTINE   : explicitsnow_age
  !!
  !>\BRIEF
  !!
  !! DESCRIPTION  : compute snow age for albedo
  !!
  !!
  !! RECENT CHANGE(S) :
  !!
  !! MAIN OUTPUT VARIABLE(S): snowage, snownobioage
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ 
!& ================================================================================================================================
  SUBROUTINE explicitsnow_age_acc(ji, kjpindex, snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, &
&snow_nobio_age)
    !$ACC ROUTINE SEQ

    IMPLICIT NONE

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: snow
    !! Snow mass [Kg/m^2]
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_snow
    !! Snowfall
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rainfall
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: temp_sol_new
    !! Surface temperature

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: snow_age
    !! Snow age
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(OUT) :: snow_nobio_age
    !! Snow age on ice, lakes, ...


    !! 0.3 Local variables
    REAL(KIND = r_std) :: d_age
    !! Snow age change
    REAL(KIND = r_std) :: xx
    !! Temporary



    !! 5.1. Snow age on land

    IF (snow(ji) .LE. zero) THEN
      snow_age(ji) = zero
    ELSE
      snow_age(ji) = (snow_age(ji) + (un - snow_age(ji) / max_snow_age) * dt_sechiba / one_day) * EXP(- precip_snow(ji) / &
&snow_trans)
    END IF

      !! 5.2. Snow age on land ice (nobio)

      !! Age of snow on ice: a little bit different because in cold regions, we really
      !! cannot negect the effect of cold temperatures on snow metamorphism any more.

      IF ((frac_snow_nobio(ji, iice) .LE. zero) .OR. (snow(ji) .LE. zero)) THEN
      snow_nobio_age(ji, iice) = zero
    ELSE

      d_age = (snow_nobio_age(ji, iice) + (un - snow_nobio_age(ji, iice) / max_snow_age) * dt_sechiba / one_day) * EXP(- &
&precip_snow(ji) / snow_trans_nobio) - snow_nobio_age(ji, iice)

        IF (d_age .GT. 0.) THEN
        xx = MAX(tp_00 - temp_sol_new(ji), zero)
        xx = (xx / omg1) ** omg2
        d_age = d_age / (un + xx)

          ! age increase more rapidly if it rains:
          IF (precip_rain(ji) .GT. 0.) THEN
          d_age = d_age * 2.
        END IF
      END IF

      snow_nobio_age(ji, iice) = MAX(snow_nobio_age(ji, iice) + d_age, zero)
    END IF

  END SUBROUTINE explicitsnow_age_acc


    !! 
!& ================================================================================================================================
    !! SUBROUTINE   : explicitsnow_age
    !!
    !>\BRIEF
    !!
    !! DESCRIPTION  : compute snow age for albedo
    !!
    !!
    !! RECENT CHANGE(S) :
    !!
    !! MAIN OUTPUT VARIABLE(S): snowage, snownobioage
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ 
!& ================================================================================================================================
    SUBROUTINE explicitsnow_age(kjpindex, snow, precip_snow, precip_rain, frac_snow_nobio, temp_sol_new, snow_age, snow_nobio_age)

    IMPLICIT NONE

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: snow
    !! Snow mass [Kg/m^2]
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_snow
    !! Snowfall
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: precip_rain
    !! Rainfall
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(IN) :: frac_snow_nobio
    !! Snow cover fraction on non-vegeted area
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: temp_sol_new
    !! Surface temperature

    !! 0.2 Output variables
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(OUT) :: snow_age
    !! Snow age
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio), INTENT(OUT) :: snow_nobio_age
    !! Snow age on ice, lakes, ...


    !! 0.3 Local variables
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: d_age
    !! Snow age change
    REAL(KIND = r_std), DIMENSION(kjpindex) :: xx
    !! Temporary


    DO ji = 1, kjpindex

        !! 5.1. Snow age on land

        IF (snow(ji) .LE. zero) THEN
        snow_age(ji) = zero
      ELSE
        snow_age(ji) = (snow_age(ji) + (un - snow_age(ji) / max_snow_age) * dt_sechiba / one_day) * EXP(- precip_snow(ji) / &
&snow_trans)
      END IF

        !! 5.2. Snow age on land ice (nobio)

        !! Age of snow on ice: a little bit different because in cold regions, we really
        !! cannot negect the effect of cold temperatures on snow metamorphism any more.

        IF ((frac_snow_nobio(ji, iice) .LE. zero) .OR. (snow(ji) .LE. zero)) THEN
        snow_nobio_age(ji, iice) = zero
      ELSE

        d_age(ji) = (snow_nobio_age(ji, iice) + (un - snow_nobio_age(ji, iice) / max_snow_age) * dt_sechiba / one_day) * EXP(- &
&precip_snow(ji) / snow_trans_nobio) - snow_nobio_age(ji, iice)

          IF (d_age(ji) .GT. 0.) THEN
          xx(ji) = MAX(tp_00 - temp_sol_new(ji), zero)
          xx(ji) = (xx(ji) / omg1) ** omg2
          d_age(ji) = d_age(ji) / (un + xx(ji))

            ! age increase more rapidly if it rains:
            IF (precip_rain(ji) .GT. 0.) THEN
            d_age(ji) = d_age(ji) * 2.
          END IF
        END IF

        snow_nobio_age(ji, iice) = MAX(snow_nobio_age(ji, iice) + d_age(ji), zero)
      END IF
    END DO

  END SUBROUTINE explicitsnow_age
  SUBROUTINE read_dummy(frac_snow_nobio, precip_rain, precip_snow, snow, temp_sol_new)
    REAL(KIND = r_std), DIMENSION(kjpindex) :: temp_sol_new
    REAL(KIND = r_std), DIMENSION(kjpindex) :: snow
    REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_snow
    REAL(KIND = r_std), DIMENSION(kjpindex) :: precip_rain
    REAL(KIND = r_std), DIMENSION(kjpindex, nnobio) :: frac_snow_nobio
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/explicitsnow_age/dummy.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- inside the read dummy routine for explicitsnow_age ---'
    READ(1363, IOSTAT = ier) frac_snow_nobio
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for frac_snow_nobio. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precip_rain
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precip_rain. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) precip_snow
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for precip_snow. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) snow
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for snow. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) temp_sol_new
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for temp_sol_new. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
  END SUBROUTINE read_dummy
END PROGRAM main
