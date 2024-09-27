
PROGRAM main
  USE module_global
  IMPLICIT NONE
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
  REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: transpir
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: humrel
  REAL(KIND = r_std), DIMENSION(kjpindex) :: evap_bare_lim
  REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: evap_bare_lim_ns
  REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_bare_soil
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm) :: e_frac
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: F_absorption
  INTEGER(KIND = i_std) :: ji
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_1
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_2
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_3
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_4
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_5
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_6
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_7
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_8
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_9
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_10
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_11
  INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_12
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption, ji, error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12)
  CALL hydrol_split_soil(kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)
  ae_ns_cpu = ae_ns
  tr_ns_cpu = tr_ns
  precisol_ns_cpu = precisol_ns
  rootsink_cpu = rootsink
  CALL initialization
  CALL read_dummy(veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption, ji, error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12)
  error_flag_hydrol_split_soil_1 = 0
  error_flag_hydrol_split_soil_2 = 0
  error_flag_hydrol_split_soil_3 = 0
  error_flag_hydrol_split_soil_4 = 0
  error_flag_hydrol_split_soil_5 = 0
  error_flag_hydrol_split_soil_6 = 0
  error_flag_hydrol_split_soil_7 = 0
  error_flag_hydrol_split_soil_8 = 0
  error_flag_hydrol_split_soil_9 = 0
  error_flag_hydrol_split_soil_10 = 0
  error_flag_hydrol_split_soil_11 = 0
  error_flag_hydrol_split_soil_12 = 0
  !$ACC ENTER DATA COPYIN(veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)
  !$ACC PARALLEL LOOP INDEPENDENT REDUCTION(+:error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12)
  DO ji = 1, kjpindex
    CALL hydrol_split_soil_acc(error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12, ji, kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(ae_ns, tr_ns, precisol_ns, rootsink)
  !$ACC EXIT DATA DELETE(veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)
  IF (ALL(ae_ns .EQ. ae_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ae_ns_gpu are equal to ae_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ae_ns_gpu do not match ae_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ae_ns - ae_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ae_ns_gpu:', MINVAL(ae_ns), MAXVAL(ae_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ae_ns_cpu:', MINVAL(ae_ns_cpu), MAXVAL(ae_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(tr_ns .EQ. tr_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in tr_ns_gpu are equal to tr_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in tr_ns_gpu do not match tr_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(tr_ns - tr_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tr_ns_gpu:', MINVAL(tr_ns), MAXVAL(tr_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of tr_ns_cpu:', MINVAL(tr_ns_cpu), MAXVAL(tr_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(precisol_ns .EQ. precisol_ns_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in precisol_ns_gpu are equal to precisol_ns_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in precisol_ns_gpu do not match precisol_ns_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(precisol_ns - precisol_ns_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of precisol_ns_gpu:', MINVAL(precisol_ns), MAXVAL(precisol_ns)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of precisol_ns_cpu:', MINVAL(precisol_ns_cpu), MAXVAL(precisol_ns_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(rootsink .EQ. rootsink_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in rootsink_gpu are equal to rootsink_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in rootsink_gpu do not match rootsink_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(rootsink - rootsink_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rootsink_gpu:', MINVAL(rootsink), MAXVAL(rootsink)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of rootsink_cpu:', MINVAL(rootsink_cpu), MAXVAL(rootsink_cpu)
    WRITE(*, *) ''
  END IF
  IF (error_flag_hydrol_split_soil_1 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_1 is > 0 :', error_flag_hydrol_split_soil_1
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'PRECISOL SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_2 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_2 is > 0 :', error_flag_hydrol_split_soil_2
  END IF
  IF (error_flag_hydrol_split_soil_3 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_3 is > 0 :', error_flag_hydrol_split_soil_3
  END IF
  IF (error_flag_hydrol_split_soil_4 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_4 is > 0 :', error_flag_hydrol_split_soil_4
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'VEVAPNU SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_5 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_5 is > 0 :', error_flag_hydrol_split_soil_5
  END IF
  IF (error_flag_hydrol_split_soil_6 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_6 is > 0 :', error_flag_hydrol_split_soil_6
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'TRANSPIR SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_7 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_7 is > 0 :', error_flag_hydrol_split_soil_7
  END IF
  IF (error_flag_hydrol_split_soil_8 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_8 is > 0 :', error_flag_hydrol_split_soil_8
  END IF
  IF (error_flag_hydrol_split_soil_9 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_9 is > 0 :', error_flag_hydrol_split_soil_9
  END IF
  IF (error_flag_hydrol_split_soil_10 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_10 is > 0 :', error_flag_hydrol_split_soil_10
    CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'ROOTSINK SPLIT FALSE')
  END IF
  IF (error_flag_hydrol_split_soil_11 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_11 is > 0 :', error_flag_hydrol_split_soil_11
  END IF
  IF (error_flag_hydrol_split_soil_12 .GT. 0) THEN
    WRITE(numout, *) 'Warning: in the hydrol_split_soil, error_flag_hydrol_split_soil_12 is > 0 :', error_flag_hydrol_split_soil_12
    CALL ipslerr_p(3, 'hydrol_split_soil', 'We will STOP now.', 'One or several fatal errors were found previously.', '')
  END IF
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_split_soil
  !!
  !>\BRIEF        Splits 2d variables into 3d variables, per soiltile (_ns suffix), at the beginning of hydrol
  !!              At this stage, the forcing fluxes to hydrol are transformed from grid-cell averages
  !!              to mean fluxes over vegtot=sum(soiltile)
  !!
  !! DESCRIPTION  :
  !! 1. Split 2d variables into 3d variables, per soiltile
  !! 1.1 Throughfall
  !! 1.2 Bare soil evaporation
  !! 1.2.2 ae_ns new
  !! 1.3 transpiration
  !! 1.4 root sink
  !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes
  !! 2.1 precisol
  !! 2.2 ae_ns and evapnu
  !! 2.3 transpiration
  !! 2.4 root sink
  !!
  !! RECENT CHANGE(S) : 2016 by A. Ducharne to match the simplification of hydrol_soil
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================


  SUBROUTINE hydrol_split_soil_acc(error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12, ji, kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)
    !$ACC ROUTINE SEQ

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_12
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_11
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_10
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_9
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_8
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_7
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_6
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_5
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_4
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_3
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_2
    INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_split_soil_1
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max Vegetation map
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soiltile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: vevapnu
    !! Bare soil evaporation
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evap_bare_lim
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: evap_bare_lim_ns
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Relative humidity per layer
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydraulic_arch = .TRUE.)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jv
    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: jst
    REAL(KIND = r_std) :: tmp_check1
    REAL(KIND = r_std) :: tmp_check2
    REAL(KIND = r_std), DIMENSION(nstm) :: tmp_check3
    LOGICAL :: error
    !_ ================================================================================================================================

    !! 1. Split 2d variables into 3d variables, per soiltile

    ! Reminders:
    !  corr_veg_soil(:,nvm,nstm) = PFT fraction per soiltile in each grid-cell
    !      corr_veg_soil(ji,jv,jst)=veget_max(ji,jv)/soiltile(ji,jst)
    !  soiltile(:,nstm) = fraction of vegtot covered by each soiltile (0-1, unitless)
    !  vegtot(:) = total fraction of grid-cell covered by PFTs (fraction with bare soil + vegetation)
    !  veget_max(:,nvm) = PFT fractions of vegtot+frac_nobio
    !  veget(:,nvm) =  fractions (of vegtot+frac_nobio) covered by vegetation in each PFT
    !       BUT veget(:,1)=veget_max(:,1)
    !  frac_bare(:,nvm) = fraction (of veget_max) with bare soil in each PFT
    !  tot_bare_soil(:) = fraction of grid mesh covered by all bare soil (=SUM(frac_bare*veget_max))
    !  frac_bare_ns(:,nstm) = evaporating bare soil fraction (of vegtot) per soiltile (defined in hydrol_vegupd)

    !! 1.1 Throughfall
    ! Transformation from precisol (flux from PFT jv in m2 of grid-mesh)
    ! to  precisol_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    precisol_ns(ji, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      IF ((veget_max(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
        precisol_ns(ji, jst) = precisol_ns(ji, jst) + precisol(ji, jv) / (soiltile(ji, jst) * vegtot(ji))
      END IF
    END DO


    !! 1.2 Bare soil evaporation and ae_ns
    ae_ns(ji, :) = zero
    DO jst = 1, nstm
      IF (evap_bare_lim(ji) .GT. min_sechiba) THEN
        ae_ns(ji, jst) = vevapnu(ji) * evap_bare_lim_ns(ji, jst) / evap_bare_lim(ji)
      END IF
    END DO

    !! 1.3 transpiration
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to tr_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    ! To do next: simplify the use of humrelv(ji,jv,jst) /humrel(ji,jv), since both are equal
    tr_ns(ji, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
        tr_ns(ji, jst) = tr_ns(ji, jst) + transpir(ji, jv) * (humrelv(ji, jv, jst) / humrel(ji, jv)) / (soiltile(ji, jst) * vegtot(ji))

      END IF
    END DO

    !! 1.4 root sink
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to root_sink (flux from contributing PFTs and soil layer with another unit, in m2 of soiltile)
    rootsink(ji, :, :) = zero

      IF (ok_hydrol_arch) THEN

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        ! OBS jst = 1,nstm
          DO jsl = 1, nslm
          IF (humrel(ji, jv) .GT. min_sechiba .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
            IF (is_tuzet_hydrol_arch) THEN
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (F_absorption(ji, jv) * e_frac(ji, jv, jsl, jst) * dt_sechiba * kilo_to_unit)
            ELSE
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (transpir(ji, jv) * e_frac(ji, jv, jsl, jst)) / (soiltile(ji, jst) * vegtot(ji))
            END IF
          END IF
        END DO
      END DO

    ELSE

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        DO jsl = 1, nslm
          IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
            rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + transpir(ji, jv) * (us(ji, jv, jst, jsl) / humrel(ji, jv)) / (soiltile(ji, jst) * vegtot(ji))
            ! rootsink(ji,1,jst)=0 as us(ji,jv,jst,1)=0
          END IF
        END DO
      END DO

    END IF
    ! ok_hydrol_arch

      !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes (grid-cell average)

      IF (check_cwrr) THEN

      error = .FALSE.

      !! 2.1 precisol

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + precisol_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO

      tmp_check2 = zero
      DO jv = 1, nvm
        tmp_check2 = tmp_check2 + precisol(ji, jv)
      END DO

        IF (ABS(tmp_check1 - tmp_check2) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_1 = error_flag_hydrol_split_soil_1 + 1
        DO jv = 1, nvm
          error_flag_hydrol_split_soil_2 = error_flag_hydrol_split_soil_2 + 1
        END DO
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_3 = error_flag_hydrol_split_soil_3 + 1
        END DO
        error = .TRUE.
      END IF

      !! 2.2 ae_ns and evapnu

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + ae_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO


        IF (ABS(tmp_check1 - vevapnu(ji)) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_4 = error_flag_hydrol_split_soil_4 + 1
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_5 = error_flag_hydrol_split_soil_5 + 1
        END DO
        error = .TRUE.
      END IF

      !! 2.3 transpiration

      tmp_check1 = zero
      DO jst = 1, nstm
        tmp_check1 = tmp_check1 + tr_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
      END DO

      tmp_check2 = zero
      DO jv = 1, nvm
        tmp_check2 = tmp_check2 + transpir(ji, jv)
      END DO

        IF (ABS(tmp_check1 - tmp_check2) .GT. allowed_err) THEN
        error_flag_hydrol_split_soil_6 = error_flag_hydrol_split_soil_6 + 1
        DO jv = 1, nvm
          error_flag_hydrol_split_soil_7 = error_flag_hydrol_split_soil_7 + 1
          DO jst = 1, nstm
            error_flag_hydrol_split_soil_8 = error_flag_hydrol_split_soil_8 + 1
          END DO
        END DO
        DO jst = 1, nstm
          error_flag_hydrol_split_soil_9 = error_flag_hydrol_split_soil_9 + 1
        END DO
        error = .TRUE.
      END IF


      !! 2.4 root sink

      tmp_check3(:) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          tmp_check3(jst) = tmp_check3(jst) + rootsink(ji, jsl, jst)
        END DO
      END DO

        DO jst = 1, nstm
        IF (ABS(tmp_check3(jst) - tr_ns(ji, jst)) .GT. allowed_err) THEN
          error_flag_hydrol_split_soil_10 = error_flag_hydrol_split_soil_10 + 1
          DO jv = 1, nvm
            error_flag_hydrol_split_soil_11 = error_flag_hydrol_split_soil_11 + 1
          END DO
          error = .TRUE.
        END IF
      END DO


        !! Exit if error was found previously in this subroutine
        IF (error) THEN
        error_flag_hydrol_split_soil_12 = error_flag_hydrol_split_soil_12 + 1
      END IF

    END IF
    ! end of check_cwrr


  END SUBROUTINE hydrol_split_soil_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_split_soil
    !!
    !>\BRIEF        Splits 2d variables into 3d variables, per soiltile (_ns suffix), at the beginning of hydrol
    !!              At this stage, the forcing fluxes to hydrol are transformed from grid-cell averages
    !!              to mean fluxes over vegtot=sum(soiltile)
    !!
    !! DESCRIPTION  :
    !! 1. Split 2d variables into 3d variables, per soiltile
    !! 1.1 Throughfall
    !! 1.2 Bare soil evaporation
    !! 1.2.2 ae_ns new
    !! 1.3 transpiration
    !! 1.4 root sink
    !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes
    !! 2.1 precisol
    !! 2.2 ae_ns and evapnu
    !! 2.3 transpiration
    !! 2.4 root sink
    !!
    !! RECENT CHANGE(S) : 2016 by A. Ducharne to match the simplification of hydrol_soil
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================


    SUBROUTINE hydrol_split_soil(kjpindex, veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption)

    !
    ! interface description

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: veget_max
    !! max Vegetation map
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: soiltile
    !! Fraction of each soiltile within vegtot (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: vevapnu
    !! Bare soil evaporation
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: transpir
    !! Transpiration
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: humrel
    !! Relative humidity
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: evap_bare_lim
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm), INTENT(IN) :: evap_bare_lim_ns
    !!
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: tot_bare_soil
    !! Total evaporating bare soil fraction
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm), INTENT(INOUT) :: us
    !! Water stress index for transpiration
    !! (by soil layer and PFT) (0-1, unitless)
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm), INTENT(IN) :: e_frac
    !! Relative humidity per layer
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: F_absorption
    !! Total root absorption (ok_hydraulic_arch = .TRUE.)

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: ji, jv, jsl, jst
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmp_check1
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tmp_check2
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: tmp_check3
    LOGICAL :: error
    !_ ================================================================================================================================

    !! 1. Split 2d variables into 3d variables, per soiltile

    ! Reminders:
    !  corr_veg_soil(:,nvm,nstm) = PFT fraction per soiltile in each grid-cell
    !      corr_veg_soil(ji,jv,jst)=veget_max(ji,jv)/soiltile(ji,jst)
    !  soiltile(:,nstm) = fraction of vegtot covered by each soiltile (0-1, unitless)
    !  vegtot(:) = total fraction of grid-cell covered by PFTs (fraction with bare soil + vegetation)
    !  veget_max(:,nvm) = PFT fractions of vegtot+frac_nobio
    !  veget(:,nvm) =  fractions (of vegtot+frac_nobio) covered by vegetation in each PFT
    !       BUT veget(:,1)=veget_max(:,1)
    !  frac_bare(:,nvm) = fraction (of veget_max) with bare soil in each PFT
    !  tot_bare_soil(:) = fraction of grid mesh covered by all bare soil (=SUM(frac_bare*veget_max))
    !  frac_bare_ns(:,nstm) = evaporating bare soil fraction (of vegtot) per soiltile (defined in hydrol_vegupd)

    !! 1.1 Throughfall
    ! Transformation from precisol (flux from PFT jv in m2 of grid-mesh)
    ! to  precisol_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    precisol_ns(:, :) = zero
    DO jv = 1, nvm
      DO ji = 1, kjpindex
        jst = pref_soil_veg(jv)
        IF ((veget_max(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
          precisol_ns(ji, jst) = precisol_ns(ji, jst) + precisol(ji, jv) / (soiltile(ji, jst) * vegtot(ji))
        END IF
      END DO
    END DO


    !! 1.2 Bare soil evaporation and ae_ns
    ae_ns(:, :) = zero
    DO jst = 1, nstm
      DO ji = 1, kjpindex
        IF (evap_bare_lim(ji) .GT. min_sechiba) THEN
          ae_ns(ji, jst) = vevapnu(ji) * evap_bare_lim_ns(ji, jst) / evap_bare_lim(ji)
        END IF
      END DO
    END DO

    !! 1.3 transpiration
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to tr_ns (flux from contributing PFTs with another unit, in m2 of soiltile)
    ! To do next: simplify the use of humrelv(ji,jv,jst) /humrel(ji,jv), since both are equal
    tr_ns(:, :) = zero
    DO jv = 1, nvm
      jst = pref_soil_veg(jv)
      DO ji = 1, kjpindex
        IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
          tr_ns(ji, jst) = tr_ns(ji, jst) + transpir(ji, jv) * (humrelv(ji, jv, jst) / humrel(ji, jv)) / (soiltile(ji, jst) * vegtot(ji))

        END IF
      END DO
    END DO

    !! 1.4 root sink
    ! Transformation from transpir (flux from PFT jv in m2 of grid-mesh)
    ! to root_sink (flux from contributing PFTs and soil layer with another unit, in m2 of soiltile)
    rootsink(:, :, :) = zero

      IF (ok_hydrol_arch) THEN

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        ! OBS jst = 1,nstm
          DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF (humrel(ji, jv) .GT. min_sechiba .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
              IF (is_tuzet_hydrol_arch) THEN
                rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (F_absorption(ji, jv) * e_frac(ji, jv, jsl, jst) * dt_sechiba * kilo_to_unit)
              ELSE
                rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + (transpir(ji, jv) * e_frac(ji, jv, jsl, jst)) / (soiltile(ji, jst) * vegtot(ji))
              END IF
            END IF
          END DO
        END DO
      END DO

    ELSE

        DO jv = 1, nvm
        jst = pref_soil_veg(jv)
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            IF ((humrel(ji, jv) .GT. min_sechiba) .AND. ((soiltile(ji, jst) * vegtot(ji)) .GT. min_sechiba)) THEN
              rootsink(ji, jsl, jst) = rootsink(ji, jsl, jst) + transpir(ji, jv) * (us(ji, jv, jst, jsl) / humrel(ji, jv)) / (soiltile(ji, jst) * vegtot(ji))
              ! rootsink(ji,1,jst)=0 as us(ji,jv,jst,1)=0
            END IF
          END DO
        END DO
      END DO

    END IF
    ! ok_hydrol_arch

      !! 2. Verification: Check if the deconvolution is correct and conserves the fluxes (grid-cell average)

      IF (check_cwrr) THEN

      error = .FALSE.

      !! 2.1 precisol

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + precisol_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

      tmp_check2(:) = zero
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          tmp_check2(ji) = tmp_check2(ji) + precisol(ji, jv)
        END DO
      END DO

        DO ji = 1, kjpindex
        IF (ABS(tmp_check1(ji) - tmp_check2(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'PRECISOL SPLIT FALSE:ji=', ji, tmp_check1(ji), tmp_check2(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - tmp_check2(ji))
          WRITE(numout, *) 'vegtot', vegtot(ji)
          DO jv = 1, nvm
            WRITE(numout, '(a,i2.2,"|",F13.4,"|",F13.4,"|",3(F9.6))') 'jv,veget_max, precisol, vegetmax_soil ', jv, veget_max(ji, jv), precisol(ji, jv), vegetmax_soil(ji, jv, :)
          END DO
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,precisol_ns', jst, precisol_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'PRECISOL SPLIT FALSE')
        END IF
      END DO

      !! 2.2 ae_ns and evapnu

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + ae_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

        DO ji = 1, kjpindex

          IF (ABS(tmp_check1(ji) - vevapnu(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'VEVAPNU SPLIT FALSE:ji, Sum(ae_ns), vevapnu =', ji, tmp_check1(ji), vevapnu(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - vevapnu(ji))
          WRITE(numout, *) 'ae_ns', ae_ns(ji, :)
          WRITE(numout, *) 'vegtot', vegtot(ji)
          WRITE(numout, *) 'evap_bare_lim, evap_bare_lim_ns', evap_bare_lim(ji), evap_bare_lim_ns(ji, :)
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,ae_ns', jst, ae_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'VEVAPNU SPLIT FALSE')
        END IF
      END DO

      !! 2.3 transpiration

      tmp_check1(:) = zero
      DO jst = 1, nstm
        DO ji = 1, kjpindex
          tmp_check1(ji) = tmp_check1(ji) + tr_ns(ji, jst) * soiltile(ji, jst) * vegtot(ji)
        END DO
      END DO

      tmp_check2(:) = zero
      DO jv = 1, nvm
        DO ji = 1, kjpindex
          tmp_check2(ji) = tmp_check2(ji) + transpir(ji, jv)
        END DO
      END DO

        DO ji = 1, kjpindex
        IF (ABS(tmp_check1(ji) - tmp_check2(ji)) .GT. allowed_err) THEN
          WRITE(numout, *) 'TRANSPIR SPLIT FALSE:ji=', ji, tmp_check1(ji), tmp_check2(ji)
          WRITE(numout, *) 'err', ABS(tmp_check1(ji) - tmp_check2(ji))
          WRITE(numout, *) 'vegtot', vegtot(ji)
          DO jv = 1, nvm
            WRITE(numout, *) 'jv,veget_max, transpir', jv, veget_max(ji, jv), transpir(ji, jv)
            DO jst = 1, nstm
              WRITE(numout, *) 'vegetmax_soil:ji,jv,jst', ji, jv, jst, vegetmax_soil(ji, jv, jst)
            END DO
          END DO
          DO jst = 1, nstm
            WRITE(numout, *) 'jst,tr_ns', jst, tr_ns(ji, jst)
            WRITE(numout, *) 'soiltile', soiltile(ji, jst)
          END DO
          error = .TRUE.
          CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'TRANSPIR SPLIT FALSE')
        END IF

      END DO

      !! 2.4 root sink

      tmp_check3(:, :) = zero
      DO jst = 1, nstm
        DO jsl = 1, nslm
          DO ji = 1, kjpindex
            tmp_check3(ji, jst) = tmp_check3(ji, jst) + rootsink(ji, jsl, jst)
          END DO
        END DO
      END DO

        DO jst = 1, nstm
        DO ji = 1, kjpindex
          IF (ABS(tmp_check3(ji, jst) - tr_ns(ji, jst)) .GT. allowed_err) THEN
            WRITE(numout, *) 'ROOTSINK SPLIT FALSE:ji,jst=', ji, jst, tmp_check3(ji, jst), tr_ns(ji, jst)
            WRITE(numout, *) 'err', ABS(tmp_check3(ji, jst) - tr_ns(ji, jst))
            WRITE(numout, *) 'HUMREL(jv=1:13)', humrel(ji, :)
            WRITE(numout, *) 'TRANSPIR', transpir(ji, :)
            DO jv = 1, nvm
              WRITE(numout, *) 'jv=', jv, 'us=', us(ji, jv, jst, :)
            END DO
            error = .TRUE.
            CALL ipslerr_p(2, 'hydrol_split_soil', 'We will STOP in the end of this subroutine.', 'check_CWRR', 'ROOTSINK SPLIT FALSE')
          END IF
        END DO
      END DO


        !! Exit if error was found previously in this subroutine
        IF (error) THEN
        WRITE(numout, *) 'One or more errors have been detected in hydrol_split_soil. Model stops.'
        CALL ipslerr_p(3, 'hydrol_split_soil', 'We will STOP now.', 'One or several fatal errors were found previously.', '')
      END IF

    END IF
    ! end of check_cwrr


  END SUBROUTINE hydrol_split_soil
  SUBROUTINE read_dummy(veget_max, soiltile, vevapnu, transpir, humrel, evap_bare_lim, evap_bare_lim_ns, tot_bare_soil, us, e_frac, F_absorption, ji, error_flag_hydrol_split_soil_1, error_flag_hydrol_split_soil_2, error_flag_hydrol_split_soil_3, error_flag_hydrol_split_soil_4, error_flag_hydrol_split_soil_5, error_flag_hydrol_split_soil_6, error_flag_hydrol_split_soil_7, error_flag_hydrol_split_soil_8, error_flag_hydrol_split_soil_9, error_flag_hydrol_split_soil_10, error_flag_hydrol_split_soil_11, error_flag_hydrol_split_soil_12)
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_12
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_11
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_10
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_9
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_8
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_7
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_6
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_5
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_4
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_3
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_2
    INTEGER(KIND = i_std) :: error_flag_hydrol_split_soil_1
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: F_absorption
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nstm) :: e_frac
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nstm, nslm) :: us
    REAL(KIND = r_std), DIMENSION(kjpindex) :: tot_bare_soil
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: evap_bare_lim_ns
    REAL(KIND = r_std), DIMENSION(kjpindex) :: evap_bare_lim
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: humrel
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: transpir
    REAL(KIND = r_std), DIMENSION(kjpindex) :: vevapnu
    REAL(KIND = r_std), DIMENSION(kjpindex, nstm) :: soiltile
    REAL(KIND = r_std), DIMENSION(kjpindex, nvm) :: veget_max
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    CALL random_number(veget_max)
    CALL random_number(soiltile)
    CALL random_number(vevapnu)
    CALL random_number(transpir)
    CALL random_number(humrel)
    CALL random_number(evap_bare_lim)
    CALL random_number(evap_bare_lim_ns)
    CALL random_number(tot_bare_soil)
    CALL random_number(us)
    CALL random_number(e_frac)
    CALL random_number(F_absorption)
    ji = 2
    error_flag_hydrol_split_soil_1 = 2
    error_flag_hydrol_split_soil_2 = 2
    error_flag_hydrol_split_soil_3 = 2
    error_flag_hydrol_split_soil_4 = 2
    error_flag_hydrol_split_soil_5 = 2
    error_flag_hydrol_split_soil_6 = 2
    error_flag_hydrol_split_soil_7 = 2
    error_flag_hydrol_split_soil_8 = 2
    error_flag_hydrol_split_soil_9 = 2
    error_flag_hydrol_split_soil_10 = 2
    error_flag_hydrol_split_soil_11 = 2
    error_flag_hydrol_split_soil_12 = 2
  END SUBROUTINE read_dummy
END PROGRAM main
