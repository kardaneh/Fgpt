
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declarations
  CALL initialization
  CALL read_dummy(ins, ji)
  CALL hydrol_soil_setup(kjpindex, ins)
  fp_cpu = fp
  g1_cpu = g1
  f_cpu = f
  ep_cpu = ep
  e_cpu = e
  gp_cpu = gp
  CALL initialization
  CALL read_dummy(ins, ji)
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_setup_acc(ji, kjpindex, ins)
  END DO
  !$ACC END PARALLEL
  !$ACC UPDATE SELF(fp, g1, f, ep, e, gp)
  IF (ALL(fp .EQ. fp_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in fp_gpu are equal to fp_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in fp_gpu do not match fp_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(fp - fp_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of fp_gpu:', MINVAL(fp), MAXVAL(fp)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of fp_cpu:', MINVAL(fp_cpu), MAXVAL(fp_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(g1 .EQ. g1_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in g1_gpu are equal to g1_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in g1_gpu do not match g1_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(g1 - g1_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of g1_gpu:', MINVAL(g1), MAXVAL(g1)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of g1_cpu:', MINVAL(g1_cpu), MAXVAL(g1_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(f .EQ. f_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in f_gpu are equal to f_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in f_gpu do not match f_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(f - f_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of f_gpu:', MINVAL(f), MAXVAL(f)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of f_cpu:', MINVAL(f_cpu), MAXVAL(f_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(ep .EQ. ep_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in ep_gpu are equal to ep_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in ep_gpu do not match ep_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(ep - ep_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ep_gpu:', MINVAL(ep), MAXVAL(ep)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of ep_cpu:', MINVAL(ep_cpu), MAXVAL(ep_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(e .EQ. e_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in e_gpu are equal to e_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in e_gpu do not match e_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(e - e_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of e_gpu:', MINVAL(e), MAXVAL(e)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of e_cpu:', MINVAL(e_cpu), MAXVAL(e_cpu)
    WRITE(*, *) ''
  END IF
  IF (ALL(gp .EQ. gp_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in gp_gpu are equal to gp_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in gp_gpu do not match gp_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(gp - gp_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of gp_gpu:', MINVAL(gp), MAXVAL(gp)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of gp_cpu:', MINVAL(gp_cpu), MAXVAL(gp_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS


  !! ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_setup
  !!
  !>\BRIEF        This subroutine computes the matrix coef.
  !!
  !! DESCRIPTION  : None
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S) : matrix coef
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_ ================================================================================================================================

  SUBROUTINE hydrol_soil_setup_acc(ji, kjpindex, ins)
    !$ACC ROUTINE SEQ


    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: temp3
    REAL(KIND = r_std) :: temp4

    !_ ================================================================================================================================
    !-we compute tridiag matrix coefficients (LEFT and RIGHT)
    ! of the system to solve [LEFT]*mc_{t+1}=[RIGHT]*mc{t}+[add terms]:
    ! e(nslm),f(nslm),g1(nslm) for the [left] vector
    ! and ep(nslm),fp(nslm),gp(nslm) for the [right] vector

    ! w_time=1 (in constantes_soil) indicates implicit computation for diffusion
    temp3 = w_time * (dt_sechiba / one_day) / deux
    temp4 = (un - w_time) * (dt_sechiba / one_day) / deux

    ! Passage to arithmetic means for layer averages also in this subroutine : Aurelien 11/05/10

    !- coefficient for first layer
    e(ji, 1) = zero
    f(ji, 1) = trois * dz(2) / huit + temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
    g1(ji, 1) = dz(2) / (huit) - temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
    ep(ji, 1) = zero
    fp(ji, 1) = trois * dz(2) / huit - temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
    gp(ji, 1) = dz(2) / (huit) + temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))

      !- coefficient for medium layers

      DO jsl = 2, nslm - 1
      e(ji, jsl) = dz(jsl) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

      f(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit + temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)))

      g1(ji, jsl) = dz(jsl + 1) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))

      ep(ji, jsl) = dz(jsl) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

      fp(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit - temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)))

      gp(ji, jsl) = dz(jsl + 1) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))
    END DO

    !- coefficient for last layer
    e(ji, nslm) = dz(nslm) / (huit) - temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
    f(ji, nslm) = trois * dz(nslm) / huit + temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * free_drain_coef(ji, ins)))
    g1(ji, nslm) = zero
    ep(ji, nslm) = dz(nslm) / (huit) + temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
    fp(ji, nslm) = trois * dz(nslm) / huit - temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * free_drain_coef(ji, ins)))
    gp(ji, nslm) = zero

  END SUBROUTINE hydrol_soil_setup_acc


    !! ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_setup
    !!
    !>\BRIEF        This subroutine computes the matrix coef.
    !!
    !! DESCRIPTION  : None
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S) : matrix coef
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_ ================================================================================================================================

    SUBROUTINE hydrol_soil_setup(kjpindex, ins)


    IMPLICIT NONE
    !
    !! 0. Variable and parameter declaration

    !! 0.1 Input variables
    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! index of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl, ji
    REAL(KIND = r_std) :: temp3, temp4

    !_ ================================================================================================================================
    !-we compute tridiag matrix coefficients (LEFT and RIGHT)
    ! of the system to solve [LEFT]*mc_{t+1}=[RIGHT]*mc{t}+[add terms]:
    ! e(nslm),f(nslm),g1(nslm) for the [left] vector
    ! and ep(nslm),fp(nslm),gp(nslm) for the [right] vector

    ! w_time=1 (in constantes_soil) indicates implicit computation for diffusion
    temp3 = w_time * (dt_sechiba / one_day) / deux
    temp4 = (un - w_time) * (dt_sechiba / one_day) / deux

      ! Passage to arithmetic means for layer averages also in this subroutine : Aurelien 11/05/10

      !- coefficient for first layer
      DO ji = 1, kjpindex
      e(ji, 1) = zero
      f(ji, 1) = trois * dz(2) / huit + temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
      g1(ji, 1) = dz(2) / (huit) - temp3 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
      ep(ji, 1) = zero
      fp(ji, 1) = trois * dz(2) / huit - temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) + a(ji, 1))
      gp(ji, 1) = dz(2) / (huit) + temp4 * ((d(ji, 1) + d(ji, 2)) / (dz(2)) - a(ji, 2))
    END DO

      !- coefficient for medium layers

      DO jsl = 2, nslm - 1
      DO ji = 1, kjpindex
        e(ji, jsl) = dz(jsl) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

        f(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit + temp3 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)))

        g1(ji, jsl) = dz(jsl + 1) / (huit) - temp3 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))

        ep(ji, jsl) = dz(jsl) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + a(ji, jsl - 1))

        fp(ji, jsl) = trois * (dz(jsl) + dz(jsl + 1)) / huit - temp4 * ((d(ji, jsl) + d(ji, jsl - 1)) / (dz(jsl)) + (d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)))

        gp(ji, jsl) = dz(jsl + 1) / (huit) + temp4 * ((d(ji, jsl) + d(ji, jsl + 1)) / (dz(jsl + 1)) - a(ji, jsl + 1))
      END DO
    END DO

      !- coefficient for last layer
      DO ji = 1, kjpindex
      e(ji, nslm) = dz(nslm) / (huit) - temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
      f(ji, nslm) = trois * dz(nslm) / huit + temp3 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * free_drain_coef(ji, ins)))
      g1(ji, nslm) = zero
      ep(ji, nslm) = dz(nslm) / (huit) + temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) + a(ji, nslm - 1))
      fp(ji, nslm) = trois * dz(nslm) / huit - temp4 * ((d(ji, nslm) + d(ji, nslm - 1)) / (dz(nslm)) - a(ji, nslm) * (un - deux * free_drain_coef(ji, ins)))
      gp(ji, nslm) = zero
    END DO

  END SUBROUTINE hydrol_soil_setup
  SUBROUTINE read_dummy(ins, ji)
    INTEGER(KIND = i_std) :: ji
    INTEGER(KIND = i_std) :: ins
    CALL random_seed(put = seed)
    WRITE(*, *) '--- inside the routine read_dummy ---'
    ins = 2
    ji = 2
  END SUBROUTINE read_dummy
END PROGRAM main
