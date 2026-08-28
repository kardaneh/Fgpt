! Copyright 2026 IPSL / CNRS / Sorbonne University
! Authors: Shivamshan Sivanesan and Kazem Ardaneh
!
! This work is licensed under the Creative Commons
! Attribution-NonCommercial-ShareAlike 4.0 International License.
! To view a copy of this license, visit
! http://creativecommons.org/licenses/by-nc-sa/4.0/

PROGRAM calibrate_w_d_multisite

  USE module_global_multilevel_matrixmodule_global_multilevel_matrix_tgt
  IMPLICIT NONE

  INTEGER :: seed_size
  INTEGER, ALLOCATABLE :: seed(:)

  INTEGER, PARAMETER :: nl = 10
  INTEGER, PARAMETER :: n_days = 365
  #ifndef N_SITES
  #define N_SITES 8
  #endif
  INTEGER, PARAMETER :: n_sites = N_SITES
  REAL(KIND=r_std), PARAMETER :: pi = 3.14159265358979_r_std

  ! Site-level fixed properties: (nlevels_tot, n_sites) and (n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: t(:,:)          ! (nlevels_tot, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: rs(:)           ! (n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: td_zero(:)
  REAL(KIND=r_std) :: mud0, rsd0

  ! Daily-varying solar geometry, per site: (n_days, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: mu(:,:)

  ! Optional per-site seasonal amplitude/offset (lets sites differ in latitude)
  REAL(KIND=r_std), ALLOCATABLE :: mu_offset(:), mu_amp(:)

  ! Parameters being calibrated (shared across all sites/days)
  REAL(KIND=r_std) :: w, d, w_true, d_true

  ! Observations: (n_days, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: fup_obs(:,:)

  ! Per-call working arrays (reused each day/site)
  REAL(KIND=r_std), ALLOCATABLE :: fup(:), fdn(:), fab(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup_w(:), fdn_w(:), fab_w(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup_d(:), fdn_d(:), fab_d(:)
  REAL(KIND=r_std), ALLOCATABLE :: fup_trial(:), fdn_trial(:), fab_trial(:)

  ! Accumulated cost / gradient / Gauss-Newton Hessian (summed over all obs)
  REAL(KIND=r_std) :: cost, cost_new
  REAL(KIND=r_std) :: gJ_w, gJ_d, grad_norm
  REAL(KIND=r_std) :: h_ww, h_wd, h_dd
  REAL(KIND=r_std) :: r_val

  ! LM damping
  REAL(KIND=r_std) :: lambda, lambda_min, lambda_max

  ! Step / line search
  REAL(KIND=r_std) :: delta_w, delta_d, delta_norm
  REAL(KIND=r_std) :: w_trial, d_trial
  REAL(KIND=r_std) :: alpha
  INTEGER :: ls_iter
  INTEGER, PARAMETER :: max_ls_iter = 20
  REAL(KIND=r_std) :: predicted_reduction, actual_reduction, rho
  REAL(KIND=r_std) :: jac_det, jac_corr

  ! Convergence
  INTEGER :: iter, max_iter
  REAL(KIND=r_std), PARAMETER :: cost_tolerance = 1.0e-14_r_std
  REAL(KIND=r_std), PARAMETER :: gradient_tolerance = 1.0e-10_r_std
  REAL(KIND=r_std), PARAMETER :: parameter_tolerance = 1.0e-9_r_std

  ! Physical bounds
  REAL(KIND=r_std), PARAMETER :: w_min = 0.0001_r_std, w_max = 1.0_r_std
  REAL(KIND=r_std), PARAMETER :: d_min = 0.0001_r_std, d_max = 10.0_r_std

  LOGICAL :: accepted, converged

  INTEGER :: j, isite   ! day index, site index

  ! Output files
  INTEGER, PARAMETER :: unit_history = 10
  INTEGER, PARAMETER :: unit_true = 11
  INTEGER, PARAMETER :: unit_fluxes = 12
  CHARACTER(LEN=100) :: history_file = 'optimization_history.txt'
  CHARACTER(LEN=100) :: true_params_file = 'true_parameters.txt'
  CHARACTER(LEN=100) :: fluxes_file = 'fluxes_comparison.txt'

  INTEGER :: n_history
  INTEGER, PARAMETER :: max_history = 1000
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_cost, history_w, history_d
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_gw, history_gd
  REAL(KIND=r_std), DIMENSION(:), ALLOCATABLE :: history_lambda, history_rho

  !
  CALL RANDOM_SEED(SIZE=seed_size)
  ALLOCATE(seed(seed_size))
  OPEN(UNIT=99, FILE='/dev/urandom', ACCESS='STREAM', FORM='UNFORMATTED', STATUS='OLD', ACTION='READ')
  READ(99) seed
  CLOSE(99)
  CALL RANDOM_SEED(PUT=seed)
  DEALLOCATE(seed)

  nlevels_tot = nl + 1

  !
  ! Allocate — everything site-indexed now
  !
  ALLOCATE(t(nlevels_tot, n_sites), rs(n_sites), td_zero(nlevels_tot))
  ALLOCATE(mu(n_days, n_sites))
  ALLOCATE(mu_offset(n_sites), mu_amp(n_sites))
  ALLOCATE(fup_obs(n_days, n_sites))

  ALLOCATE(fup(nlevels_tot), fdn(nlevels_tot), fab(nlevels_tot))
  ALLOCATE(fup_w(nlevels_tot), fdn_w(nlevels_tot), fab_w(nlevels_tot))
  ALLOCATE(fup_d(nlevels_tot), fdn_d(nlevels_tot), fab_d(nlevels_tot))
  ALLOCATE(fup_trial(nlevels_tot), fdn_trial(nlevels_tot), fab_trial(nlevels_tot))

  ALLOCATE(history_cost(max_history), history_w(max_history), history_d(max_history))
  ALLOCATE(history_gw(max_history), history_gd(max_history))
  ALLOCATE(history_lambda(max_history), history_rho(max_history))

  n_history = 0
  td_zero = 0.0_r_std
  mud0    = 0.0_r_std
  rsd0    = 0.0_r_std

  !
  ! Per-site fixed properties: optical depth profile + surface reflectance
  ! Each site gets its own random canopy structure and soil reflectance.
  !
  DO isite = 1, n_sites
     CALL RANDOM_NUMBER(t(:,isite))
     ! vary the depth-scale range a bit per site so they're not identical
     t(:,isite) = 0.1_r_std + (2.0_r_std + REAL(isite,r_std) - 0.1_r_std) * t(:,isite)
     CALL RANDOM_NUMBER(rs(isite))

     ! per-site seasonal solar-geometry parameters (e.g. different latitudes)
     CALL RANDOM_NUMBER(mu_offset(isite))
     mu_offset(isite) = 0.45_r_std + 0.15_r_std * mu_offset(isite)   ! ~0.45-0.60
     CALL RANDOM_NUMBER(mu_amp(isite))
     mu_amp(isite) = 0.20_r_std + 0.20_r_std * mu_amp(isite)          ! ~0.20-0.40
  END DO

  !
  ! Daily solar geometry per site: mu = cos(solar zenith angle at noon)
  !
  DO isite = 1, n_sites
     DO j = 1, n_days
        mu(j,isite) = mu_offset(isite) + mu_amp(isite) * &
             COS(2.0_r_std*pi*(REAL(j,r_std) - 172.0_r_std)/365.0_r_std)
     END DO
  END DO

  !
  ! True parameters (synthetic twin experiment) — shared across all sites
  !
  CALL RANDOM_NUMBER(w_true)
  CALL RANDOM_NUMBER(d_true)
  d_true = 0.1_r_std + (3.0_r_std - 0.1_r_std) * d_true

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  WRITE(*,'(A,I0,A)') ' ', n_sites, '-SITE, FULL-YEAR (365 noon obs) CALIBRATION'
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  WRITE(*,'(A,F12.8,A,F12.8)') 'True parameters: w = ', w_true, ', d = ', d_true

  OPEN(unit_true, FILE=true_params_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_true, '(A,ES18.10)') 'True_w = ', w_true
  WRITE(unit_true, '(A,ES18.10)') 'True_d = ', d_true
  CLOSE(unit_true)
  WRITE(*,'(A)') 'True parameters written to: ' // TRIM(true_params_file)

  !
  ! Generate synthetic observations: fup(nlevels_tot) per day, per site
  !
  DO isite = 1, n_sites
     DO j = 1, n_days
        CALL multilevel_matrix(nl, mu(j,isite), rs(isite), t(:,isite), &
             w_true, d_true, fup, fdn, fab)
        fup_obs(j,isite) = fup(nlevels_tot)
     END DO
  END DO

  WRITE(*,'(A,I0,A)') 'Generated ', n_sites*n_days, ' synthetic observations.'

  !
  ! Initial guess
  !
  w = 0.5_r_std
  d = 0.5_r_std
  w = MIN(MAX(w, w_min), w_max)
  d = MIN(MAX(d, d_min), d_max)

  lambda     = 1.0e-3_r_std
  lambda_min = 1.0e-12_r_std
  lambda_max = 1.0e12_r_std
  max_iter   = 200
  converged  = .FALSE.

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  WRITE(*,'(A,I0,A)') ' LEVENBERG-MARQUARDT OPTIMIZATION (', n_sites, ' sites x 365 days)'
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  WRITE(*,'(A,F12.8,A,F12.8)') 'Initial guess: w = ', w, ', d = ', d
  WRITE(*,'(A)') ''

  !
  ! OPTIMIZATION LOOP
  !
  DO iter = 1, max_iter

     cost = 0.0_r_std
     gJ_w = 0.0_r_std
     gJ_d = 0.0_r_std
     h_ww = 0.0_r_std
     h_wd = 0.0_r_std
     h_dd = 0.0_r_std

     ! --- accumulate over ALL sites and ALL days ---
     DO isite = 1, n_sites
        DO j = 1, n_days

           CALL multilevel_matrix_d( &
                nl, mu(j,isite), mud0, rs(isite), rsd0, t(:,isite), td_zero, &
                w, 1.0_r_std, d, 0.0_r_std, &
                fup, fup_w, fdn, fdn_w, fab, fab_w)

           CALL multilevel_matrix_d( &
                nl, mu(j,isite), mud0, rs(isite), rsd0, t(:,isite), td_zero, &
                w, 0.0_r_std, d, 1.0_r_std, &
                fup, fup_d, fdn, fdn_d, fab, fab_d)

           r_val = fup(nlevels_tot) - fup_obs(j,isite)

           cost = cost + 0.5_r_std * r_val**2
           gJ_w = gJ_w + r_val * fup_w(nlevels_tot)
           gJ_d = gJ_d + r_val * fup_d(nlevels_tot)
           h_ww = h_ww + fup_w(nlevels_tot)**2
           h_wd = h_wd + fup_w(nlevels_tot) * fup_d(nlevels_tot)
           h_dd = h_dd + fup_d(nlevels_tot)**2

        END DO
     END DO

     grad_norm = SQRT(gJ_w**2 + gJ_d**2)

     IF (h_ww > 1.0e-30_r_std .AND. h_dd > 1.0e-30_r_std) THEN
        jac_corr = h_wd / SQRT(h_ww * h_dd)
     ELSE
        jac_corr = 0.0_r_std
     END IF

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

     ! --- Solve damped Gauss-Newton (LM) 2x2 system ---
     jac_det = (h_ww + lambda) * (h_dd + lambda) - h_wd**2
     IF (ABS(jac_det) < 1.0e-30_r_std) THEN
        lambda = MIN(10.0_r_std * lambda, lambda_max)
        WRITE(*,'(A,ES12.4)') 'Near-singular Hessian. Increasing lambda to ', lambda
        CYCLE
     END IF

     delta_w = (-(h_dd + lambda) * gJ_w + h_wd * gJ_d) / jac_det
     delta_d = ( h_wd * gJ_w - (h_ww + lambda) * gJ_d) / jac_det
     delta_norm = SQRT(delta_w**2 + delta_d**2)

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

     ! --- Backtracking line search (plain forward model, all sites) ---
     alpha = 1.0_r_std
     accepted = .FALSE.

     DO ls_iter = 1, max_ls_iter

        w_trial = MIN(MAX(w + alpha * delta_w, w_min), w_max)
        d_trial = MIN(MAX(d + alpha * delta_d, d_min), d_max)

        cost_new = 0.0_r_std

        DO isite = 1, n_sites
           DO j = 1, n_days
              CALL multilevel_matrix(nl, mu(j,isite), rs(isite), t(:,isite), &
                   w_trial, d_trial, fup_trial, fdn_trial, fab_trial)
              cost_new = cost_new + 0.5_r_std * (fup_trial(nlevels_tot) - fup_obs(j,isite))**2
           END DO
        END DO

        IF (cost_new < cost) THEN
           accepted = .TRUE.
           EXIT
        END IF

        alpha = 0.5_r_std * alpha
     END DO

     IF (accepted) THEN

        actual_reduction = cost - cost_new
        predicted_reduction = &
             -(gJ_w * (alpha * delta_w) + gJ_d * (alpha * delta_d)) &
             - 0.5_r_std * ( &
               h_ww * (alpha * delta_w)**2 + &
               2.0_r_std * h_wd * (alpha * delta_w) * (alpha * delta_d) + &
               h_dd * (alpha * delta_d)**2 )

        IF (predicted_reduction > 1.0e-30_r_std) THEN
           rho = actual_reduction / predicted_reduction
        ELSE
           rho = 0.0_r_std
        END IF

        w = w_trial
        d = d_trial
        cost = cost_new

        IF (rho > 0.75_r_std) THEN
           lambda = MAX(lambda / 3.0_r_std, lambda_min)
        ELSE IF (rho < 0.25_r_std) THEN
           lambda = MIN(lambda * 4.0_r_std, lambda_max)
        END IF
        IF (alpha < 1.0_r_std) lambda = MIN(lambda * 2.0_r_std, lambda_max)

     ELSE
        lambda = MIN(lambda * 10.0_r_std, lambda_max)
        rho = -1.0_r_std
     END IF

     WRITE(*,'(A,I4,A,ES12.4,A,F10.6,A,F10.6,A,ES10.3,A,ES10.3,A,F8.4,A,F8.4)') &
          'iter=', iter, ' cost=', cost, ' w=', w, ' d=', d, &
          ' lambda=', lambda, ' alpha=', alpha, ' rho=', rho

     IF (ABS(w - w_true) < 1.0e-6_r_std .AND. ABS(d - d_true) < 1.0e-6_r_std) THEN
        WRITE(*,'(A)') 'Recovered the true parameters.'
        converged = .TRUE.
        EXIT
     END IF

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

  !
  ! Results
  !
  WRITE(*,'(A)') ''
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  WRITE(*,'(A)') ' OPTIMIZATION COMPLETE'
  WRITE(*,'(A)') '-------------------------------------------------------------------'
  IF (converged) THEN
     WRITE(*,'(A)') 'Status: CONVERGED'
  ELSE
     WRITE(*,'(A)') 'Status: MAXIMUM ITERATIONS REACHED'
  END IF
  WRITE(*,'(A,F12.8)') 'True w       = ', w_true
  WRITE(*,'(A,F12.8)') 'Recovered w  = ', w
  WRITE(*,'(A,ES14.6)') 'Absolute error w = ', ABS(w - w_true)
  WRITE(*,'(A,F12.8)') 'True d       = ', d_true
  WRITE(*,'(A,F12.8)') 'Recovered d  = ', d
  WRITE(*,'(A,ES14.6)') 'Absolute error d = ', ABS(d - d_true)
  WRITE(*,'(A,ES18.10)') 'Final cost = ', cost

  jac_det = h_ww*h_dd - h_wd**2
  IF (jac_det > 1.0e-30_r_std) THEN
     WRITE(*,'(A,F10.6)') 'Approx. std. error on w = ', SQRT(h_dd/jac_det)
     WRITE(*,'(A,F10.6)') 'Approx. std. error on d = ', SQRT(h_ww/jac_det)
  END IF

  OPEN(unit_history, FILE=history_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_history, '(A)') '# Optimization History'
  WRITE(unit_history, '(A)') '# Columns: iter cost w d gJ_w gJ_d lambda rho'
  DO iter = 1, n_history
     WRITE(unit_history, '(I6,7(ES18.10))') &
        iter, history_cost(iter), history_w(iter), history_d(iter), &
        history_gw(iter), history_gd(iter), history_lambda(iter), history_rho(iter)
  END DO
  CLOSE(unit_history)
  WRITE(*,'(A)') 'Optimization history written to: ' // TRIM(history_file)

  ! Final flux comparison, all sites
  OPEN(unit_fluxes, FILE=fluxes_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_fluxes, '(A)') '# Flux Comparison (Timeseries, multi-site)'
  WRITE(unit_fluxes, '(A)') '# Columns: site day fup_model fup_obs'
  DO isite = 1, n_sites
     DO j = 1, n_days
        CALL multilevel_matrix(nl, mu(j,isite), rs(isite), t(:,isite), w, d, fup, fdn, fab)
        WRITE(unit_fluxes, '(2I6,2(ES18.10))') isite, j, fup(nlevels_tot), fup_obs(j,isite)
     END DO
  END DO
  CLOSE(unit_fluxes)
  WRITE(*,'(A)') 'Flux comparison written to: ' // TRIM(fluxes_file)

  DEALLOCATE(t, rs, td_zero, mu, mu_offset, mu_amp, fup_obs)
  DEALLOCATE(fup, fdn, fab, fup_w, fdn_w, fab_w, fup_d, fdn_d, fab_d)
  DEALLOCATE(fup_trial, fdn_trial, fab_trial)
  DEALLOCATE(history_cost, history_w, history_d)
  DEALLOCATE(history_gw, history_gd, history_lambda, history_rho)

END PROGRAM calibrate_w_d_multisite
