
PROGRAM main
  USE module_global
  IMPLICIT NONE
  INTEGER(KIND = i_std) :: ins
  INTEGER(KIND = i_std) :: ji
  WRITE(*, *) '--- inside the main program ---'
  CALL declaration_initialization
  CALL read_dummy(ins)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  CALL hydrol_soil_tridiag(kjpindex, ins)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  mcl_cpu = mcl
  CALL declaration_initialization
  CALL read_dummy(ins)
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  start_time = ic0 * 1.0 / icr
  !$ACC PARALLEL LOOP INDEPENDENT
  DO ji = 1, kjpindex
    CALL hydrol_soil_tridiag_acc(ji, kjpindex, ins)
  END DO
  !$ACC END PARALLEL
  CALL SYSTEM_CLOCK(ic0, icr, ic)
  stop_time = ic0 * 1.0 / icr
  WRITE(*, *) "Execution time : ", stop_time - start_time
  !$ACC UPDATE SELF(mcl)
  IF (ALL(mcl .EQ. mcl_cpu)) THEN
    WRITE(*, *) 'Test passed: All elements in mcl_gpu are equal to mcl_cpu.'
  ELSE
    WRITE(*, *) ''
    WRITE(*, *) 'Test failed: All elements in mcl_gpu do not match mcl_cpu.'
    WRITE(*, '(A, E25.16)') 'Maximum absolute error:', MAXVAL(ABS(mcl - mcl_cpu))
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_gpu:', MINVAL(mcl), MAXVAL(mcl)
    WRITE(*, '(A, 2E25.16)') 'Min and Max of mcl_cpu:', MINVAL(mcl_cpu), MAXVAL(mcl_cpu)
    WRITE(*, *) ''
  END IF
  CONTAINS


  !!
  !& 
!& ================================================================================================================================
  !! SUBROUTINE   : hydrol_soil_tridiag
  !!
  !>\BRIEF        This subroutine solves a set of linear equations which has a tridiagonal coefficient matrix.
  !!
  !! DESCRIPTION  : It is only applied in the grid-cells where resolv(ji)=TRUE
  !!
  !! RECENT CHANGE(S) : None
  !!
  !! MAIN OUTPUT VARIABLE(S) : mcl (global module variable)
  !!
  !! REFERENCE(S) :
  !!
  !! FLOWCHART    : None
  !! \n
  !_
  !& 
!& ================================================================================================================================
  !_ hydrol_soil_tridiag

  SUBROUTINE hydrol_soil_tridiag_acc(ji, kjpindex, ins)
    !$ACC ROUTINE SEQ

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    INTEGER(KIND = i_std), INTENT(IN) :: ji
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! number of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    REAL(KIND = r_std) :: bet
    REAL(KIND = r_std), DIMENSION(nslm) :: gam

    !_
    !& 
!& ================================================================================================================================

    IF (resolv(ji)) THEN
      bet = tmat(ji, 1, 2)
      mcl(ji, 1, ins) = rhs(ji, 1) / bet
    END IF

      DO jsl = 2, nslm

        IF (resolv(ji)) THEN

        gam(jsl) = tmat(ji, jsl - 1, 3) / bet
        bet = tmat(ji, jsl, 2) - tmat(ji, jsl, 1) * gam(jsl)
        mcl(ji, jsl, ins) = (rhs(ji, jsl) - tmat(ji, jsl, 1) * mcl(ji, jsl - 1, ins)) / bet
      END IF

    END DO

      IF (resolv(ji)) THEN
      DO jsl = nslm - 1, 1, - 1
        mcl(ji, jsl, ins) = mcl(ji, jsl, ins) - gam(jsl + 1) * mcl(ji, jsl + 1, ins)
      END DO
    END IF

  END SUBROUTINE hydrol_soil_tridiag_acc


    !!
    !& 
!& ================================================================================================================================
    !! SUBROUTINE   : hydrol_soil_tridiag
    !!
    !>\BRIEF        This subroutine solves a set of linear equations which has a tridiagonal coefficient matrix.
    !!
    !! DESCRIPTION  : It is only applied in the grid-cells where resolv(ji)=TRUE
    !!
    !! RECENT CHANGE(S) : None
    !!
    !! MAIN OUTPUT VARIABLE(S) : mcl (global module variable)
    !!
    !! REFERENCE(S) :
    !!
    !! FLOWCHART    : None
    !! \n
    !_
    !& 
!& ================================================================================================================================
    !_ hydrol_soil_tridiag

    SUBROUTINE hydrol_soil_tridiag(kjpindex, ins)

    !- arguments

    !! 0. Variable and parameter declaration

    !! 0.1 Input variables

    INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
    !! Domain size
    INTEGER(KIND = i_std), INTENT(IN) :: ins
    !! number of soil type

    !! 0.2 Output variables

    !! 0.3 Modified variables

    !! 0.4 Local variables

    INTEGER(KIND = i_std) :: jsl
    INTEGER(KIND = i_std) :: ji
    REAL(KIND = r_std), DIMENSION(kjpindex) :: bet
    REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: gam

    !_
    !& 
!& ================================================================================================================================
    DO ji = 1, kjpindex

        IF (resolv(ji)) THEN
        bet(ji) = tmat(ji, 1, 2)
        mcl(ji, 1, ins) = rhs(ji, 1) / bet(ji)
      END IF
    END DO

      DO jsl = 2, nslm
      DO ji = 1, kjpindex

          IF (resolv(ji)) THEN

          gam(ji, jsl) = tmat(ji, jsl - 1, 3) / bet(ji)
          bet(ji) = tmat(ji, jsl, 2) - tmat(ji, jsl, 1) * gam(ji, jsl)
          mcl(ji, jsl, ins) = (rhs(ji, jsl) - tmat(ji, jsl, 1) * mcl(ji, jsl - 1, ins)) / bet(ji)
        END IF

      END DO
    END DO

      DO ji = 1, kjpindex
      IF (resolv(ji)) THEN
        DO jsl = nslm - 1, 1, - 1
          mcl(ji, jsl, ins) = mcl(ji, jsl, ins) - gam(ji, jsl + 1) * mcl(ji, jsl + 1, ins)
        END DO
      END IF
    END DO

  END SUBROUTINE hydrol_soil_tridiag
  SUBROUTINE read_dummy(ins)
    INTEGER(KIND = i_std) :: ins
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_soil_tridiag/dummy.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- inside the read dummy routine for hydrol_soil_tridiag ---'
    READ(1363, IOSTAT = ier) ins
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ins. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
  END SUBROUTINE read_dummy
END PROGRAM main
