! Copyright 2026 IPSL / CNRS / Sorbonne University
! Authors: Shivamshan Sivanesan and Kazem Ardaneh
!
! This work is licensed under the Creative Commons
! Attribution-NonCommercial-ShareAlike 4.0 International License.
! To view a copy of this license, visit
! http://creativecommons.org/licenses/by-nc-sa/4.0/

PROGRAM calibrate_w_d

  USE module_global_multilevel_matrixmodule_global_multilevel_matrix_tgt
  IMPLICIT NONE

  INTEGER :: seed_size
  INTEGER, ALLOCATABLE :: seed(:)

  ! Problem definition

  INTEGER, PARAMETER :: nl = 10

  REAL(KIND=r_std) :: mu1, mu2
  REAL(KIND=r_std) :: rs1, rs2
  REAL(KIND=r_std) :: mud0, rsd0

  REAL(KIND=r_std) :: w, d
  REAL(KIND=r_std) :: w_true, d_true

  REAL(KIND=r_std), ALLOCATABLE :: t1(:), t2(:)
  REAL(KIND=r_std), ALLOCATABLE :: td_zero(:)

  ! Arrays : site 1

  REAL(KIND=r_std), ALLOCATABLE :: fup1(:), fdn1(:), fab1(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup1_w(:), fdn1_w(:), fab1_w(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup1_d(:), fdn1_d(:), fab1_d(:)

  ! Arrays : site 2

  REAL(KIND=r_std), ALLOCATABLE :: fup2(:), fdn2(:), fab2(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup2_w(:), fdn2_w(:), fab2_w(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup2_d(:), fdn2_d(:), fab2_d(:)

  ! References: Albedo values at the top of the structures at the two sites

  REAL(KIND=r_std) :: fup_obs_1
  REAL(KIND=r_std) :: fup_obs_2

  ! Trial arrays

  REAL(KIND=r_std), ALLOCATABLE :: fup_trial(:)
  REAL(KIND=r_std), ALLOCATABLE :: fdn_trial(:)
  REAL(KIND=r_std), ALLOCATABLE :: fab_trial(:)

  ! Residuals

  REAL(KIND=r_std) :: r1
  REAL(KIND=r_std) :: r2

  ! Optimization

  REAL(KIND=r_std) :: cost
  REAL(KIND=r_std) :: cost_new

  REAL(KIND=r_std) :: gJ_w
  REAL(KIND=r_std) :: gJ_d
  REAL(KIND=r_std) :: grad_norm

  ! Gauss-Newton Hessian
  REAL(KIND=r_std) :: h_ww
  REAL(KIND=r_std) :: h_wd
  REAL(KIND=r_std) :: h_dd

  ! LM damping
  REAL(KIND=r_std) :: lambda
  REAL(KIND=r_std) :: lambda_min
  REAL(KIND=r_std) :: lambda_max

  ! Parameter update
  REAL(KIND=r_std) :: delta_w
  REAL(KIND=r_std) :: delta_d
  REAL(KIND=r_std) :: delta_norm

  ! Trial parameters
  REAL(KIND=r_std) :: w_trial
  REAL(KIND=r_std) :: d_trial

  ! Line search
  REAL(KIND=r_std) :: alpha
  INTEGER :: ls_iter
  INTEGER, PARAMETER :: max_ls_iter = 20

  ! Reduction
  REAL(KIND=r_std) :: predicted_reduction
  REAL(KIND=r_std) :: actual_reduction
  REAL(KIND=r_std) :: rho

  ! Jacobian diagnostics
  REAL(KIND=r_std) :: jac_det
  REAL(KIND=r_std) :: jac_corr

  ! Convergence
  INTEGER :: iter
  INTEGER :: max_iter

  REAL(KIND=r_std), PARAMETER :: cost_tolerance = 1.0e-14_r_std
  REAL(KIND=r_std), PARAMETER :: gradient_tolerance = 1.0e-10_r_std
  REAL(KIND=r_std), PARAMETER :: parameter_tolerance = 1.0e-9_r_std

  ! Physical bounds
  REAL(KIND=r_std), PARAMETER :: w_min = 0.0001_r_std
  REAL(KIND=r_std), PARAMETER :: w_max = 1.0_r_std

  REAL(KIND=r_std), PARAMETER :: d_min = 0.0001_r_std
  REAL(KIND=r_std), PARAMETER :: d_max = 10.0_r_std

  LOGICAL :: accepted
  LOGICAL :: converged

  ! Files

  INTEGER, PARAMETER :: unit_history = 10
  INTEGER, PARAMETER :: unit_fluxes = 11
  INTEGER, PARAMETER :: unit_true = 12

  CHARACTER(LEN=100) :: history_file = 'optimization_history.txt'
  CHARACTER(LEN=100) :: fluxes_file = 'fluxes_comparison.txt'
  CHARACTER(LEN=100) :: true_params_file = 'true_parameters.txt'

  ! History

  INTEGER :: n_history
  INTEGER, PARAMETER :: max_history = 1000
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_cost
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_w
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_d
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_gw
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_gd
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_lambda
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_rho

  ! Random seed

  CALL RANDOM_SEED(SIZE=seed_size)
  ALLOCATE(seed(seed_size))
  OPEN(UNIT=99, FILE='/dev/urandom', ACCESS='STREAM', FORM='UNFORMATTED', STATUS='OLD', ACTION='READ')
  READ(99) seed
  CLOSE(99)
  CALL RANDOM_SEED(PUT=seed)
  DEALLOCATE(seed)

  ! Number of levels

  nlevels_tot = nl + 1

  ! Allocate

  ALLOCATE(t1(nlevels_tot))
  ALLOCATE(t2(nlevels_tot))
  ALLOCATE(td_zero(nlevels_tot))

  ALLOCATE(fup1(nlevels_tot))
  ALLOCATE(fdn1(nlevels_tot))
  ALLOCATE(fab1(nlevels_tot))

  ALLOCATE(fup1_w(nlevels_tot))
  ALLOCATE(fdn1_w(nlevels_tot))
  ALLOCATE(fab1_w(nlevels_tot))

  ALLOCATE(fup1_d(nlevels_tot))
  ALLOCATE(fdn1_d(nlevels_tot))
  ALLOCATE(fab1_d(nlevels_tot))

  ALLOCATE(fup2(nlevels_tot))
  ALLOCATE(fdn2(nlevels_tot))
  ALLOCATE(fab2(nlevels_tot))

  ALLOCATE(fup2_w(nlevels_tot))
  ALLOCATE(fdn2_w(nlevels_tot))
  ALLOCATE(fab2_w(nlevels_tot))

  ALLOCATE(fup2_d(nlevels_tot))
  ALLOCATE(fdn2_d(nlevels_tot))
  ALLOCATE(fab2_d(nlevels_tot))

  ALLOCATE(fup_trial(nlevels_tot))
  ALLOCATE(fdn_trial(nlevels_tot))
  ALLOCATE(fab_trial(nlevels_tot))

  ALLOCATE(history_cost(max_history))
  ALLOCATE(history_w(max_history))
  ALLOCATE(history_d(max_history))
  ALLOCATE(history_gw(max_history))
  ALLOCATE(history_gd(max_history))
  ALLOCATE(history_lambda(max_history))
  ALLOCATE(history_rho(max_history))

  n_history = 0

  ! Set model parameters

  mud0 = 0.0_r_std
  rsd0 = 0.0_r_std

  ! site 1

  CALL RANDOM_NUMBER(t1)
  t1 = 0.1_r_std + (8.0_r_std - 0.1_r_std) * t1
  CALL RANDOM_NUMBER(mu1)
  CALL RANDOM_NUMBER(rs1)

  ! site 2

  CALL RANDOM_NUMBER(t2)
  t2 = 0.1_r_std + (3.0_r_std - 0.1_r_std) * t2
  CALL RANDOM_NUMBER(mu2)
  CALL RANDOM_NUMBER(rs2)

  td_zero = 0.0_r_std

  ! Ref parameters

  CALL RANDOM_NUMBER(w_true)
  CALL RANDOM_NUMBER(d_true)
  d_true = 0.1_r_std + (3.0_r_std - 0.1_r_std) * d_true

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ' Two-site CALIBRATION'
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A,F12.8,A,F12.8)') 'True parameters: w = ', w_true, ', d = ', d_true
  WRITE(*,'(A)') ''
  WRITE(*,'(A,F12.8)') 'site 1: mu = ', mu1
  WRITE(*,'(A,F12.8)') 'site 1: rs = ', rs1
  WRITE(*,'(A)') ''
  WRITE(*,'(A,F12.8)') 'site 2: mu = ', mu2
  WRITE(*,'(A,F12.8)') 'site 2: rs = ', rs2

  OPEN(unit_true, FILE=true_params_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_true, '(A,ES18.10)') 'True_w = ', w_true
  WRITE(unit_true, '(A,ES18.10)') 'True_d = ', d_true
  CLOSE(unit_true)
  WRITE(*,'(A)') 'True parameters written to: ' // TRIM(true_params_file)

  ! References

  w = w_true
  d = d_true

  CALL multilevel_matrix_d( &
       nl, mu1, mud0, rs1, rsd0, t1, td_zero, &
       w, 0.0_r_std, d, 0.0_r_std, &
       fup1, fup1_w, fdn1, fdn1_w, fab1, fab1_w)

  CALL multilevel_matrix_d( &
       nl, mu2, mud0, rs2, rsd0, t2, td_zero, &
       w, 0.0_r_std, d, 0.0_r_std, &
       fup2, fup2_w, fdn2, fdn2_w, fab2, fab2_w)

  fup_obs_1 = fup1(nlevels_tot)
  fup_obs_2 = fup2(nlevels_tot)

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ' SYNTHETIC REFERENCES'
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A,ES18.10)') 'site 1 fup(nlevels_tot) = ', fup_obs_1
  WRITE(*,'(A,ES18.10)') 'site 2 fup(nlevels_tot) = ', fup_obs_2

  ! Initial guess

  w = 0.5_r_std
  d = 0.5_r_std

  w = MIN(MAX(w, w_min), w_max)
  d = MIN(MAX(d, d_min), d_max)

  ! LM parameters

  lambda     = 1.0e-3_r_std
  lambda_min = 1.0e-12_r_std
  lambda_max = 1.0e12_r_std

  max_iter = 200
  converged = .FALSE.

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ' LEVENBERG-MARQUARDT OPTIMIZATION'
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A,F12.8,A,F12.8)') 'Initial guess: w = ', w, ', d = ', d
  WRITE(*,'(A)') ''

  ! OPTIMIZATION LOOP

  DO iter = 1, max_iter

     ! site 1 derivative with respect to w

     CALL multilevel_matrix_d( &
          nl, mu1, mud0, rs1, rsd0, t1, td_zero, &
          w, 1.0_r_std, d, 0.0_r_std, &
          fup1, fup1_w, fdn1, fdn1_w, fab1, fab1_w)

     ! site 1 derivative with respect to d

     CALL multilevel_matrix_d( &
          nl, mu1, mud0, rs1, rsd0, t1, td_zero, &
          w, 0.0_r_std, d, 1.0_r_std, &
          fup1, fup1_d, fdn1, fdn1_d, fab1, fab1_d)

     ! site 2 derivative with respect to w

     CALL multilevel_matrix_d( &
          nl, mu2, mud0, rs2, rsd0, t2, td_zero, &
          w, 1.0_r_std, d, 0.0_r_std, &
          fup2, fup2_w, fdn2, fdn2_w, fab2, fab2_w)

     ! site 2 derivative with respect to d

     CALL multilevel_matrix_d( &
          nl, mu2, mud0, rs2, rsd0, t2, td_zero, &
          w, 0.0_r_std, d, 1.0_r_std, &
          fup2, fup2_d, fdn2, fdn2_d, fab2, fab2_d)

     ! RESIDUALS

     r1 = fup1(nlevels_tot) - fup_obs_1
     r2 = fup2(nlevels_tot) - fup_obs_2

     ! COST

     cost = 0.5_r_std * (r1**2 + r2**2)

     ! GRADIENT
     !
     ! J =
     !
     ! [ dfup1/dw   dfup1/dd ]
     ! [ dfup2/dw   dfup2/dd ]
     !

     gJ_w = r1 * fup1_w(nlevels_tot) + r2 * fup2_w(nlevels_tot)
     gJ_d = r1 * fup1_d(nlevels_tot) + r2 * fup2_d(nlevels_tot)
     grad_norm = SQRT(gJ_w**2 + gJ_d**2)


     ! GAUSS-NEWTON HESSIAN

     h_ww = fup1_w(nlevels_tot)**2 + fup2_w(nlevels_tot)**2
     h_wd = fup1_w(nlevels_tot) * fup1_d(nlevels_tot) + fup2_w(nlevels_tot) * fup2_d(nlevels_tot)
     h_dd = fup1_d(nlevels_tot)**2 + fup2_d(nlevels_tot)**2

     ! Jacobian correlation

     IF (h_ww > 1.0e-30_r_std .AND. h_dd > 1.0e-30_r_std) THEN
        jac_corr = h_wd / SQRT(h_ww * h_dd)
     ELSE
        jac_corr = 0.0_r_std
     END IF

     ! Convergence checks

     IF (cost < cost_tolerance) THEN
        WRITE(*,'(A)') 'Converged: cost tolerance reached.'
        converged = .TRUE.
        EXIT
     END IF

     IF (grad_norm < gradient_tolerance) THEN
        WRITE(*,'(A)') 'Converged: gradient tolerance reached.'
        converged = .TRUE.
        EXIT
     END IF

     ! Solve LM system
     !
     ! [hww+l   hwd ] [dw] = [-gw]
     ! [hwd   hdd+l] [dd]   [-gd]

     jac_det = (h_ww + lambda) * (h_dd + lambda) - h_wd**2

     IF (ABS(jac_det) < 1.0e-30_r_std) THEN
        lambda = MIN(10.0_r_std * lambda, lambda_max)
        WRITE(*,'(A,ES12.4)') 'Near-singular Hessian. Increasing lambda to ', lambda
        CYCLE
     END IF

     delta_w = (-(h_dd + lambda) * gJ_w + h_wd * gJ_d) / jac_det
     delta_d = ( h_wd * gJ_w - (h_ww + lambda) * gJ_d) / jac_det
     delta_norm = SQRT(delta_w**2 + delta_d**2)

     ! Parameter convergence

     IF (delta_norm < parameter_tolerance * (1.0_r_std + SQRT(w**2 + d**2))) THEN
        IF (grad_norm < gradient_tolerance) THEN
           WRITE(*,'(A)') 'Converged: small parameter step and gradient.'
           converged = .TRUE.
           EXIT
        ELSE
           lambda = MIN(10.0_r_std * lambda, lambda_max)
           CYCLE
        END IF
     END IF

     ! Backtracking line search

     alpha = 1.0_r_std
     accepted = .FALSE.

     DO ls_iter = 1, max_ls_iter

        w_trial = w + alpha * delta_w
        d_trial = d + alpha * delta_d

        ! Apply physical bounds

        w_trial = MIN(MAX(w_trial, w_min), w_max)
        d_trial = MIN(MAX(d_trial, d_min), d_max)

        ! Trial site 1

        CALL multilevel_matrix_d( &
             nl, mu1, mud0, rs1, rsd0, t1, td_zero, &
             w_trial, 0.0_r_std, d_trial, 0.0_r_std, &
             fup_trial, fup1_w, fdn_trial, fdn1_w, &
             fab_trial, fab1_w)

        ! Save site 1 trial value

        cost_new = 0.5_r_std * (fup_trial(nlevels_tot) - fup_obs_1)**2

        ! Trial site 2

        CALL multilevel_matrix_d( &
             nl, mu2, mud0, rs2, rsd0, t2, td_zero, &
             w_trial, 0.0_r_std, d_trial, 0.0_r_std, &
             fup_trial, fup1_w, fdn_trial, fdn1_w, &
             fab_trial, fab1_w)

        ! Add site 2 contribution

        cost_new = cost_new + 0.5_r_std * (fup_trial(nlevels_tot) - fup_obs_2)**2

        ! Accept if cost decreases

        IF (cost_new < cost) THEN
           accepted = .TRUE.
           EXIT
        END IF

        alpha = 0.5_r_std * alpha
     END DO

     ! Handle accepted/rejected step

     IF (accepted) THEN

        actual_reduction = cost - cost_new
        predicted_reduction = &
             -(gJ_w * (alpha * delta_w) + &
               gJ_d * (alpha * delta_d)) &
             - 0.5_r_std * ( &
               h_ww * (alpha * delta_w)**2 + &
               2.0_r_std * h_wd * &
               (alpha * delta_w) * &
               (alpha * delta_d) + &
               h_dd * (alpha * delta_d)**2 )

        IF (predicted_reduction > 1.0e-30_r_std) THEN
           rho = actual_reduction / predicted_reduction
        ELSE
           rho = 0.0_r_std
        END IF

        ! Accept parameters

        w = w_trial
        d = d_trial
        cost = cost_new

        ! Update damping

        IF (rho > 0.75_r_std) THEN
           lambda = MAX(lambda / 3.0_r_std, lambda_min)
        ELSE IF (rho < 0.25_r_std) THEN
           lambda = MIN(lambda * 4.0_r_std, lambda_max)
        END IF

        IF (alpha < 1.0_r_std) THEN
           lambda = MIN(lambda * 2.0_r_std, lambda_max)
        END IF

     ELSE
        lambda = MIN(lambda * 10.0_r_std, lambda_max)
        rho = -1.0_r_std
     END IF

     ! Progress
     WRITE(*,'(A,I4,A,ES12.4,A,F10.6,A,F10.6, &
          A,ES10.3,A,ES10.3,A,ES10.3,A,F8.4,A,F8.4)') &
          'iter=', iter, &
          ' cost=', cost, &
          ' w=', w, &
          ' d=', d, &
          ' gw=', gJ_w, &
          ' gd=', gJ_d, &
          ' lambda=', lambda, &
          ' alpha=', alpha, &
          ' rho=', rho

     ! Check true parameter recovery

     IF (ABS(w - w_true) < 1.0e-6_r_std .AND. ABS(d - d_true) < 1.0e-6_r_std) THEN
        WRITE(*,'(A)') ''
        WRITE(*,'(A)') 'Recovered the true parameters.'
        converged = .TRUE.
        EXIT
     END IF

     ! Warning

     IF (MOD(iter,20) == 0) THEN
        WRITE(*,'(A,ES12.4)') 'Jacobian correlation = ', jac_corr
        IF (ABS(jac_corr) > 0.999_r_std) THEN
           WRITE(*,'(A)') 'WARNING: w and d may be poorly identifiable.'
        END IF
     END IF

     n_history = n_history + 1
     history_cost(n_history) = cost
     history_w(n_history) = w
     history_d(n_history) = d
     history_gw(n_history) = gJ_w
     history_gd(n_history) = gJ_d
     history_lambda(n_history) = lambda
     IF (accepted) THEN
          history_rho(n_history) = rho
     ELSE
          history_rho(n_history) = -1.0_r_std
     END IF

  END DO

  ! Final forward calculations

  CALL multilevel_matrix_d( &
       nl, mu1, mud0, rs1, rsd0, t1, td_zero, &
       w, 0.0_r_std, d, 0.0_r_std, &
       fup1, fup1_w, fdn1, fdn1_w, fab1, fab1_w)

  CALL multilevel_matrix_d( &
       nl, mu2, mud0, rs2, rsd0, t2, td_zero, &
       w, 0.0_r_std, d, 0.0_r_std, &
       fup2, fup2_w, fdn2, fdn2_w, fab2, fab2_w)


  ! Results

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ' OPTIMIZATION COMPLETE'
  WRITE(*,'(A)') '============================================================'

  IF (converged) THEN
     WRITE(*,'(A)') 'Status: CONVERGED'
  ELSE
     WRITE(*,'(A)') 'Status: MAXIMUM ITERATIONS REACHED'
  END IF


  WRITE(*,'(A)') ''
  WRITE(*,'(A,F12.8)') 'True w       = ', w_true
  WRITE(*,'(A,F12.8)') 'Recovered w  = ', w
  WRITE(*,'(A,ES14.6)') 'Absolute error w = ', ABS(w - w_true)
  WRITE(*,'(A)') ''
  WRITE(*,'(A,F12.8)') 'True d       = ', d_true
  WRITE(*,'(A,F12.8)') 'Recovered d  = ', d
  WRITE(*,'(A,ES14.6)') 'Absolute error d = ', ABS(d - d_true)
  WRITE(*,'(A)') ''
  WRITE(*,'(A,ES18.10)') 'Final cost = ', cost
  WRITE(*,'(A,ES18.10)') 'Final gradient norm = ', grad_norm
  WRITE(*,'(A,ES18.10)') 'Final lambda = ', lambda


  ! Final comparison

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ' FINAL reference COMPARISON'
  WRITE(*,'(A)') '============================================================'
  WRITE(*,'(A)') ''
  WRITE(*,'(A)') 'site 1:'
  WRITE(*,'(A,ES18.10)')'  fup model = ', fup1(nlevels_tot)
  WRITE(*,'(A,ES18.10)')'  fup obs   = ', fup_obs_1
  WRITE(*,'(A,F12.6,A)')'  relative error = ', 100.0_r_std * ABS(fup1(nlevels_tot) - fup_obs_1) / MAX(ABS(fup_obs_1),1.0e-12_r_std), '%'
  WRITE(*,'(A)') ''
  WRITE(*,'(A)') 'site 2:'
  WRITE(*,'(A,ES18.10)')'  fup model = ', fup2(nlevels_tot)
  WRITE(*,'(A,ES18.10)')'  fup obs   = ', fup_obs_2
  WRITE(*,'(A,F12.6,A)')'  relative error = ', 100.0_r_std * ABS(fup2(nlevels_tot) - fup_obs_2) / MAX(ABS(fup_obs_2),1.0e-12_r_std), '%'

  ! History to file

  OPEN(unit_history, FILE=history_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_history, '(A)') '# Optimization History'
  WRITE(unit_history, '(A)') '# Columns: iter cost w d gJ_w gJ_d lambda rho'
  DO iter = 1, n_history
     WRITE(unit_history, '(I6,7(ES18.10))') &
        iter, &
        history_cost(iter), &
        history_w(iter), &
        history_d(iter), &
        history_gw(iter), &
        history_gd(iter), &
        history_lambda(iter), &
        history_rho(iter)
  END DO
  CLOSE(unit_history)
  WRITE(*,'(A)') 'Optimization history written to: ' // TRIM(history_file)

  DEALLOCATE(t1)
  DEALLOCATE(t2)
  DEALLOCATE(td_zero)

  DEALLOCATE(fup1)
  DEALLOCATE(fdn1)
  DEALLOCATE(fab1)

  DEALLOCATE(fup1_w)
  DEALLOCATE(fdn1_w)
  DEALLOCATE(fab1_w)

  DEALLOCATE(fup1_d)
  DEALLOCATE(fdn1_d)
  DEALLOCATE(fab1_d)

  DEALLOCATE(fup2)
  DEALLOCATE(fdn2)
  DEALLOCATE(fab2)

  DEALLOCATE(fup2_w)
  DEALLOCATE(fdn2_w)
  DEALLOCATE(fab2_w)

  DEALLOCATE(fup2_d)
  DEALLOCATE(fdn2_d)
  DEALLOCATE(fab2_d)

  DEALLOCATE(fup_trial)
  DEALLOCATE(fdn_trial)
  DEALLOCATE(fab_trial)

END PROGRAM calibrate_w_d
