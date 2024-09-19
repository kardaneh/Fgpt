
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
  REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, njsc, mcr, mcs, ji)
  CALL hydrol_soil_coef(mcr, mcs, kjpindex, ins, njsc)
  d_cpu = d
  a_cpu = a
  k_cpu = k
  b_cpu = b
  CALL initialization
  CALL read_dummy(ins, njsc, mcr, mcs, ji)
  !$ACC ENTER DATA COPYIN(mcr, mcs, njsc)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, ins, njsc)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(d, a, k, b)
  !$ACC EXIT DATA DELETE(mcr, mcs, njsc)
  IF (ALL(d .EQ. d_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in d_gpu are equal to d_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in d_gpu do not match d_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(d - d_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of d_gpu:', MINVAL(d), MAXVAL(d)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of d_cpu:', MINVAL(d_cpu), MAXVAL(d_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(a .EQ. a_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in a_gpu are equal to a_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in a_gpu do not match a_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(a - a_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of a_gpu:', MINVAL(a), MAXVAL(a)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of a_cpu:', MINVAL(a_cpu), MAXVAL(a_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(k .EQ. k_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in k_gpu are equal to k_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in k_gpu do not match k_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(k - k_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of k_gpu:', MINVAL(k), MAXVAL(k)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of k_cpu:', MINVAL(k_cpu), MAXVAL(k_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(b .EQ. b_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in b_gpu are equal to b_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in b_gpu do not match b_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(b - b_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of b_gpu:', MINVAL(b), MAXVAL(b)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of b_cpu:', MINVAL(b_cpu), MAXVAL(b_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_coef
  !!
  !>\BRIEF        Computes coef for the linearised hydraulic conductivity
  !! k_lin=a_lin mc_lin+b_lin and the linearised diffusivity d_lin.
  !!
  !! DESCRIPTION  :
  !! First, we identify the interval i in which the current value of mc is located.
  !! Then, we give the values of the linearized parameters to compute
  !! conductivity and diffusivity as K=a*mc+b and d.
  !!
  !! RECENT CHANGE(S) : Addition of the dependence to profil_froz_hydro_ns
  !!
  !! MAIN OUTPUT VARIABLE(S) :
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================
  !_ hydrol_soil_coef

  SUBROUTINE hydrol_soil_coef_acc(ji, mcr, mcs, kjpindex, ins, njsc)
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
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: i
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: mc_used
    !! Used liquid water content
    REAL(KIND = r_std) :: x
    REAL(KIND = r_std) :: m

    !_ ================================================================================================================================

    IF (ok_freeze_cwrr) THEN

        ! Calculation of liquid and frozen saturation degrees with respect to residual
        ! x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
        ! 1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)

        DO jsl = 1, nslm

        x = 1._r_std - profil_froz_hydro_ns(ji, jsl, ins)

        ! mc_used is used in the calculation of hydrological properties
        ! It corresponds to a liquid mc, but the expression is different from mcl in hydrol_soil,
        ! to ensure that we get the a, b, d of the first bin when mcl<mcr
        mc_used = mcr(ji) + x * MAX((mc(ji, jsl, ins) - mcr(ji)), zero)
        !
        ! calcul de k based on mc_liq
        !
        i = MAX(imin, MIN(imax - 1, INT(imin + (imax - imin) * (mc_used - mcr(ji)) / (mcs(ji) - mcr(ji)))))
        a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm^2/d
        k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc_used + b_lin(i, jsl, ji))
        ! in mm/d
        ! loop on grid
      END DO

    ELSE
      ! .NOT. ok_freeze_cwrr
        DO jsl = 1, nslm

        ! it is impossible to consider a mc<mcr for the binning
        mc_ratio = MAX(mc(ji, jsl, ins) - mcr(ji), zero) / (mcs(ji) - mcr(ji))

        i = MAX(MIN(INT((imax - imin) * mc_ratio) + imin, imax - 1), imin)
        a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm/d
        d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
        ! in mm^2/d
        k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc(ji, jsl, ins) + b_lin(i, jsl, ji))
        ! in mm/d
      END DO
    END IF

  END SUBROUTINE hydrol_soil_coef_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_coef
    !!
    !>\BRIEF        Computes coef for the linearised hydraulic conductivity
    !! k_lin=a_lin mc_lin+b_lin and the linearised diffusivity d_lin.
    !!
    !! DESCRIPTION  :
    !! First, we identify the interval i in which the current value of mc is located.
    !! Then, we give the values of the linearized parameters to compute
    !! conductivity and diffusivity as K=a*mc+b and d.
    !!
    !! RECENT CHANGE(S) : Addition of the dependence to profil_froz_hydro_ns
    !!
    !! MAIN OUTPUT VARIABLE(S) :
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================
    !_ hydrol_soil_coef

    SUBROUTINE hydrol_soil_coef(mcr, mcs, kjpindex, ins, njsc)

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
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
    !! Residual volumetric water content (m^{3} m^{-3})
    REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
    !! Saturated volumetric water content (m^{3} m^{-3})

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl, ji, i
    REAL(KIND = r_std) :: mc_ratio
    REAL(KIND = r_std) :: mc_used
    !! Used liquid water content
    REAL(KIND = r_std) :: x, m

    !_ ================================================================================================================================

    IF (ok_freeze_cwrr) THEN

        ! Calculation of liquid and frozen saturation degrees with respect to residual
        ! x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
        ! 1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)

        DO jsl = 1, nslm
        DO ji = 1, kjpindex

          x = 1._r_std - profil_froz_hydro_ns(ji, jsl, ins)

          ! mc_used is used in the calculation of hydrological properties
          ! It corresponds to a liquid mc, but the expression is different from mcl in hydrol_soil,
          ! to ensure that we get the a, b, d of the first bin when mcl<mcr
          mc_used = mcr(ji) + x * MAX((mc(ji, jsl, ins) - mcr(ji)), zero)
          !
          ! calcul de k based on mc_liq
          !
          i = MAX(imin, MIN(imax - 1, INT(imin + (imax - imin) * (mc_used - mcr(ji)) / (mcs(ji) - mcr(ji)))))
          a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm^2/d
          k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc_used + b_lin(i, jsl, ji))
          ! in mm/d
        END DO
        ! loop on grid
      END DO

    ELSE
      ! .NOT. ok_freeze_cwrr
        DO jsl = 1, nslm
        DO ji = 1, kjpindex

          ! it is impossible to consider a mc<mcr for the binning
          mc_ratio = MAX(mc(ji, jsl, ins) - mcr(ji), zero) / (mcs(ji) - mcr(ji))

          i = MAX(MIN(INT((imax - imin) * mc_ratio) + imin, imax - 1), imin)
          a(ji, jsl) = a_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          b(ji, jsl) = b_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm/d
          d(ji, jsl) = d_lin(i, jsl, ji) * kfact_root(ji, jsl, ins)
          ! in mm^2/d
          k(ji, jsl) = kfact_root(ji, jsl, ins) * MAX(k_lin(imin + 1, jsl, ji), a_lin(i, jsl, ji) * mc(ji, jsl, ins) + b_lin(i, jsl, ji))
          ! in mm/d
        END DO
      END DO
    END IF

  END SUBROUTINE hydrol_soil_coef
  SUBROUTINE read_dummy(ins, njsc, mcr, mcs, ji)
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcs
    REAL(KIND = r_std), DIMENSION(kjpindex) :: mcr
    INTEGER(KIND = i_std), DIMENSION(kjpindex) :: njsc
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    njsc = 2
    CALL random_number(mcr)
    CALL random_number(mcs)
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
