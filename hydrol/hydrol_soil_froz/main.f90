
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
  INTEGER(KIND = i_std) :: ins
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declaration_initialization
  CALL read_dummy(avan, ins, mcr, mcs, njsc, nvan, stempdiag)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  CALL hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  profil_froz_hydro_ns_cpu = profil_froz_hydro_ns
  CALL declaration_initialization
  CALL read_dummy(avan, ins, mcr, mcs, njsc, nvan, stempdiag)
  !$ACC ENTER DATA COPYIN(nvan, avan, mcr, mcs, njsc, stempdiag)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_froz_acc(ji, nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)
  END DO
  !$ACC END PARALLEL
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  !$ACC UPDATE SELF(profil_froz_hydro_ns)
  !$ACC EXIT DATA DELETE(nvan, avan, mcr, mcs, njsc, stempdiag)
  IF (ALL(profil_froz_hydro_ns .EQ. profil_froz_hydro_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in profil_froz_hydro_ns_gpu are equal to profil_froz_hydro_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in profil_froz_hydro_ns_gpu do not match profil_froz_hydro_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(profil_froz_hydro_ns - profil_froz_hydro_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_ns_gpu:', MINVAL(profil_froz_hydro_ns), MAXVAL(profil_froz_hydro_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of profil_froz_hydro_ns_cpu:', MINVAL(profil_froz_hydro_ns_cpu), &
&MAXVAL(profil_froz_hydro_ns_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS

  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_froz
  !!
  !>\BRIEF        Computes profil_froz_hydro_ns, the fraction of frozen water in the soil layers.
  !!
  !! DESCRIPTION  :
  !!
  !! RECENT CHANGE(S) : Created by A. Ducharne in 2016.
  !!
  !! MAIN OUTPUT VARIABLE(S) : profil_froz_hydro_ns
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_froz

  SUBROUTINE hydrol_soil_froz_acc(ji, nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)
    !$ACC ROUTINE SEQ

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x
    REAL(KIND = r_std) :: denom
    REAL(KIND = r_std) :: froz_frac_moy
    REAL(KIND = r_std) :: smtot_moy
    REAL(KIND = r_std), DIMENSION(nslm) :: mc_ns

    !_
    !& 
!& ================================================================================================================================

    !    ONLY FOR THE (ok_freeze_cwrr) CASE

    ! Calculation of liquid and frozen saturation degrees above residual moisture
    !   x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
    !   1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)
    ! It's important for the good work of the water diffusion scheme (tridiag) that the total
    ! liquid water also includes mcr, so mcl > 0 even when x=0

    DO jsl = 1, nslm
      ! Van Genuchten parameter for thermodynamical calculation
      m = 1. - 1. / nvan(ji)

        IF ((.NOT. ok_thermodynamical_freezing) .OR. (mc(ji, jsl, ins) .LT. (mcr(ji) + min_sechiba))) THEN
        ! Linear soil freezing or soil moisture below residual
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          x = (stempdiag(ji, jsl) - (fr_center - fr_dT / 2.)) / fr_dT
        ELSE
          x = 0._r_std
        END IF
      ELSE IF (ok_thermodynamical_freezing) THEN
        ! Thermodynamical soil freezing
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          ! Factor 2.2 from the PhD of Isabelle Gouttevin
          x = MIN(((mcs(ji) - mcr(ji)) * ((2.2 * 1000. * avan(ji) * (fr_center + fr_dT / 2. - stempdiag(ji, jsl)) * lhf / &
&ZeroCelsius / 10.) ** nvan(ji) + 1.) ** (- m)) / (mc(ji, jsl, ins) - mcr(ji)), 1._r_std)
        ELSE
          x = 0._r_std
        END IF
      END IF

      profil_froz_hydro_ns(ji, jsl, ins) = 1._r_std - x

      mc_ns(jsl) = mc(ji, jsl, ins) / mcs(ji)

      ! loop on grid
    END DO

    ! Applay correction on the frozen fraction
    ! Depends on two external parameters: froz_frac_corr and smtot_corr
    froz_frac_moy = zero
    denom = zero
    DO jsl = 1, nslm
      froz_frac_moy = froz_frac_moy + dh(jsl) * profil_froz_hydro_ns(ji, jsl, ins)
      denom = denom + dh(jsl)
    END DO
    froz_frac_moy = froz_frac_moy / denom

    smtot_moy = zero
    denom = zero
    DO jsl = 1, nslm - 1
      smtot_moy = smtot_moy + dh(jsl) * mc_ns(jsl)
      denom = denom + dh(jsl)
    END DO
    smtot_moy = smtot_moy / denom

      DO jsl = 1, nslm
      profil_froz_hydro_ns(ji, jsl, ins) = MIN(profil_froz_hydro_ns(ji, jsl, ins) * (froz_frac_moy ** froz_frac_corr) * (smtot_moy &
&** smtot_corr), max_froz_hydro)
    END DO

  END SUBROUTINE hydrol_soil_froz_acc

    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_froz
    !!
    !>\BRIEF        Computes profil_froz_hydro_ns, the fraction of frozen water in the soil layers.
    !!
    !! DESCRIPTION  :
    !!
    !! RECENT CHANGE(S) : Created by A. Ducharne in 2016.
    !!
    !! MAIN OUTPUT VARIABLE(S) : profil_froz_hydro_ns
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_froz

    SUBROUTINE hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)

    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! Index of soil type
    INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
    !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
    !! Van Genuchten coeficients n (unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
    !! Van Genuchten coeficients a (mm-1})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
    !! Diagnostic temp profile from thermosoil

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: i
    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: m
    REAL(KIND = r_std) :: x
    REAL(KIND = r_std) :: denom
    REAL(KIND = r_std), DIMENSION(kjpindex) :: froz_frac_moy
    REAL(KIND = r_std), DIMENSION(kjpindex) :: smtot_moy
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mc_ns

    !_
    !& 
!& ================================================================================================================================

    !    ONLY FOR THE (ok_freeze_cwrr) CASE

    ! Calculation of liquid and frozen saturation degrees above residual moisture
    !   x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
    !   1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)
    ! It's important for the good work of the water diffusion scheme (tridiag) that the total
    ! liquid water also includes mcr, so mcl > 0 even when x=0

    DO jsl = 1, nslm
      DO ji = 1, kjpindex
        ! Van Genuchten parameter for thermodynamical calculation
        m = 1. - 1. / nvan(ji)

          IF ((.NOT. ok_thermodynamical_freezing) .OR. (mc(ji, jsl, ins) .LT. (mcr(ji) + min_sechiba))) THEN
          ! Linear soil freezing or soil moisture below residual
            IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
            x = 1._r_std
          ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
            x = (stempdiag(ji, jsl) - (fr_center - fr_dT / 2.)) / fr_dT
          ELSE
            x = 0._r_std
          END IF
        ELSE IF (ok_thermodynamical_freezing) THEN
          ! Thermodynamical soil freezing
            IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
            x = 1._r_std
          ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
            ! Factor 2.2 from the PhD of Isabelle Gouttevin
            x = MIN(((mcs(ji) - mcr(ji)) * ((2.2 * 1000. * avan(ji) * (fr_center + fr_dT / 2. - stempdiag(ji, jsl)) * lhf / &
&ZeroCelsius / 10.) ** nvan(ji) + 1.) ** (- m)) / (mc(ji, jsl, ins) - mcr(ji)), 1._r_std)
          ELSE
            x = 0._r_std
          END IF
        END IF

        profil_froz_hydro_ns(ji, jsl, ins) = 1._r_std - x

        mc_ns(ji, jsl) = mc(ji, jsl, ins) / mcs(ji)

      END DO
      ! loop on grid
    END DO

    ! Applay correction on the frozen fraction
    ! Depends on two external parameters: froz_frac_corr and smtot_corr
    froz_frac_moy(:) = zero
    denom = zero
    DO jsl = 1, nslm
      froz_frac_moy(:) = froz_frac_moy(:) + dh(jsl) * profil_froz_hydro_ns(:, jsl, ins)
      denom = denom + dh(jsl)
    END DO
    froz_frac_moy(:) = froz_frac_moy(:) / denom

    smtot_moy(:) = zero
    denom = zero
    DO jsl = 1, nslm - 1
      smtot_moy(:) = smtot_moy(:) + dh(jsl) * mc_ns(:, jsl)
      denom = denom + dh(jsl)
    END DO
    smtot_moy(:) = smtot_moy(:) / denom

      DO jsl = 1, nslm
      profil_froz_hydro_ns(:, jsl, ins) = MIN(profil_froz_hydro_ns(:, jsl, ins) * (froz_frac_moy(:) ** froz_frac_corr) * &
&(smtot_moy(:) ** smtot_corr), max_froz_hydro)
    END DO

  END SUBROUTINE hydrol_soil_froz
  SUBROUTINE read_dummy(avan, ins, mcr, mcs, njsc, nvan, stempdiag)
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: stempdiag
    REAL(KIND = r_std), DIMENSION(kjpindex) :: nvan
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    INTEGER(KIND = i_std) :: ins
    REAL(KIND = r_std), DIMENSION(kjpindex) :: avan
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_froz/dummy.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- inside the read dummy routine for hydrol_soil_froz ---'
    READ(1363, IOSTAT = ier) avan
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for avan. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ins
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ins. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcr. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcs
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcs. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) njsc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for njsc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) nvan
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for nvan. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) stempdiag
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for stempdiag. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
  END SUBROUTINE read_dummy
END PROGRAM main
