! Copyright 2026 IPSL / CNRS / Sorbonne University
! Authors: Shivamshan Sivanesan and Kazem Ardaneh
!
! This work is licensed under the Creative Commons
! Attribution-NonCommercial-ShareAlike 4.0 International License.
! To view a copy of this license, visit
! http://creativecommons.org/licenses/by-nc-sa/4.0/

PROGRAM calibrate_w_d_rs

  USE module_global_multilevel_matrixmodule_global_multilevel_matrix_adj
  IMPLICIT NONE

  INTEGER, PARAMETER :: nl = 10
  INTEGER, PARAMETER :: n_days = 365

  #ifndef N_SITES
  #define N_SITES 8
  #endif

  INTEGER, PARAMETER :: n_sites = N_SITES
  INTEGER, PARAMETER :: np = n_sites + 2   ! w, d, rs(1:n_sites)

  REAL(KIND=r_std), PARAMETER :: pi = 3.14159265358979_r_std

  ! Site-level fixed properties: (nlevels_tot, n_sites) and (n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: t(:,:)        ! (nlevels_tot, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: rs_true(:)    ! (n_sites)
  REAL(KIND=r_std) :: w_true, d_true

  ! Daily-varying solar geometry, per site: (n_days, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: mu(:,:)
  REAL(KIND=r_std), ALLOCATABLE :: mu_offset(:)
  ! Optional per-site seasonal amplitude/offset (lets sites differ in latitude)
  REAL(KIND=r_std), ALLOCATABLE :: mu_amp(:)

  ! Observations: (n_days, n_sites)
  REAL(KIND=r_std), ALLOCATABLE :: fup_obs(:,:)

  ! Forward-model variables
  REAL(KIND=r_std), ALLOCATABLE :: fup(:)
  REAL(KIND=r_std), ALLOCATABLE :: fdn(:)
  REAL(KIND=r_std), ALLOCATABLE :: fab(:)

  ! Adjoint variables
  REAL(KIND=r_std), ALLOCATABLE :: fupb(:)
  REAL(KIND=r_std), ALLOCATABLE :: fdnb(:)
  REAL(KIND=r_std), ALLOCATABLE :: fabb(:)
  REAL(KIND=r_std), ALLOCATABLE :: tb(:)
  REAL(KIND=r_std) :: wb, db, mub, rsb

  ! Parameter vector: p(1)=w, p(2)=d, p(2+isite)=rs(isite)
  REAL(KIND=r_std), DIMENSION(np) :: p, p_true, p_trial
  REAL(KIND=r_std), DIMENSION(np) :: grad_p, row_grad, delta_p
  REAL(KIND=r_std), DIMENSION(np) :: p_min, p_max
  REAL(KIND=r_std), DIMENSION(np,np) :: Hess, Hess_lm

  ! Optimization quantities
  REAL(KIND=r_std) :: cost, cost_new
  REAL(KIND=r_std) :: grad_norm
  REAL(KIND=r_std) :: step_norm
  REAL(KIND=r_std) :: rho, predicted_reduction, actual_reduction

  REAL(KIND=r_std) :: lambda_lm, lambda_lm_min, lambda_lm_max

  INTEGER :: iter, max_iter, ls_iter_lm
  INTEGER, PARAMETER :: max_ls_iter = 100

  LOGICAL :: accepted, converged, singular

  ! Convergence tolerances
  REAL(KIND=r_std) :: gradient_tolerance
  REAL(KIND=r_std), PARAMETER :: parameter_tolerance = 1.0e-10_r_std
  REAL(KIND=r_std), PARAMETER :: cost_tolerance = 1.0e-16_r_std

  ! Random seed
  INTEGER :: seed_size
  INTEGER, ALLOCATABLE :: seed(:)

  ! Loop variables
  INTEGER :: isite, j, k

  ! Diagnostics
  REAL(KIND=r_std) :: max_gradient
  REAL(KIND=r_std) :: old_cost

  ! Output files
  INTEGER, PARAMETER :: unit_true = 10
  INTEGER, PARAMETER :: unit_history = 11
  INTEGER, PARAMETER :: unit_profiles = 12
  INTEGER, PARAMETER :: unit_fluxes = 13

  CHARACTER(LEN=100) :: true_file    = 'w_d_rs_true.txt'
  CHARACTER(LEN=100) :: history_file = 'w_d_rs_optimization_history.txt'
  CHARACTER(LEN=100) :: profile_file = 'w_d_rs_recovered.txt'
  CHARACTER(LEN=100) :: flux_file    = 'w_d_rs_flux_comparison.txt'

  nlevels_tot = nl + 1

  ! Random seed
  CALL RANDOM_SEED(SIZE=seed_size)
  ALLOCATE(seed(seed_size))
  OPEN(UNIT=99, FILE='/dev/urandom', ACCESS='STREAM', FORM='UNFORMATTED', STATUS='OLD', ACTION='READ')
  READ(99) seed
  CLOSE(99)
  CALL RANDOM_SEED(PUT=seed)
  DEALLOCATE(seed)

  !
  ! Allocate arrays
  !

  ALLOCATE(t(nlevels_tot,n_sites))
  ALLOCATE(rs_true(n_sites))
  ALLOCATE(mu(n_days,n_sites))
  ALLOCATE(mu_offset(n_sites))
  ALLOCATE(mu_amp(n_sites))
  ALLOCATE(fup_obs(n_days,n_sites))

  ALLOCATE(fup(nlevels_tot))
  ALLOCATE(fdn(nlevels_tot))
  ALLOCATE(fab(nlevels_tot))
  ALLOCATE(fupb(nlevels_tot))
  ALLOCATE(fdnb(nlevels_tot))
  ALLOCATE(fabb(nlevels_tot))
  ALLOCATE(tb(nlevels_tot))

  !
  ! Generate site properties
  !

  DO isite = 1, n_sites

     ! Fixed, known optical depth profile (not calibrated)
     CALL RANDOM_NUMBER(t(:,isite))
     t(:,isite) = 0.2_r_std + 1.8_r_std * t(:,isite)

     ! True surface reflectance (to be recovered)
     CALL RANDOM_NUMBER(rs_true(isite))
     rs_true(isite) = 0.05_r_std + 0.35_r_std * rs_true(isite)

     ! Solar geometry: offset > amp by construction so mu stays
     ! smoothly varying and positive all year.
     CALL RANDOM_NUMBER(mu_offset(isite))
     mu_offset(isite) = 0.45_r_std + 0.35_r_std * mu_offset(isite)

     CALL RANDOM_NUMBER(mu_amp(isite))
     mu_amp(isite) = 0.5_r_std * mu_offset(isite) * &
          (0.4_r_std + 0.4_r_std * mu_amp(isite))

  END DO

  DO isite = 1, n_sites
     DO j = 1, n_days
        mu(j,isite) = mu_offset(isite) + mu_amp(isite) * &
             COS(2.0_r_std*pi*(REAL(j,r_std)-172.0_r_std)/365.0_r_std)
        mu(j,isite) = MAX(0.05_r_std, mu(j,isite))
     END DO
  END DO

  !
  ! True shared parameters w, d
  !

  CALL RANDOM_NUMBER(w_true)
  w_true = 0.5_r_std + 0.45_r_std * w_true      ! in (0.5, 0.95)

  CALL RANDOM_NUMBER(d_true)
  d_true = 0.5_r_std + 2.0_r_std * d_true       ! in (0.5, 2.5)

  !
  ! Pack true parameter vector
  !

  p_true(1) = w_true
  p_true(2) = d_true
  DO isite = 1, n_sites
     p_true(2+isite) = rs_true(isite)
  END DO

  !
  ! Bounds
  !

  p_min(1) = 0.001_r_std;  p_max(1) = 0.999_r_std   ! w
  p_min(2) = 0.001_r_std;  p_max(2) = 10.0_r_std    ! d
  DO isite = 1, n_sites
     p_min(2+isite) = 0.0_r_std
     p_max(2+isite) = 0.99_r_std                    ! rs(isite)
  END DO

  !
  ! Print experiment information
  !

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') ''
  WRITE(*,'(A)') ' JOINT w, d, rs(site) CALIBRATION (ADJOINT + LEVENBERG-MARQUARDT)'
  WRITE(*,'(A)') ''
  WRITE(*,'(A,I0)') 'Number of sites          = ', n_sites
  WRITE(*,'(A,I0)') 'Number of days           = ', n_days
  WRITE(*,'(A,I0)') 'Number of parameters     = ', np
  WRITE(*,'(A,F12.6)') 'True w                   = ', w_true
  WRITE(*,'(A,F12.6)') 'True d                   = ', d_true
  WRITE(*,'(A)') ''

  OPEN(unit_true, FILE=true_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_true,'(A,ES18.10)') 'True_w = ', w_true
  WRITE(unit_true,'(A,ES18.10)') 'True_d = ', d_true
  DO isite = 1, n_sites
     WRITE(unit_true,'(A,I0,A,ES18.10)') 'True_rs(', isite, ') = ', rs_true(isite)
  END DO
  CLOSE(unit_true)

  !
  ! Generate synthetic references using TRUE parameters
  !

  WRITE(*,'(A)') 'Generating synthetic references...'
  DO isite = 1, n_sites
     DO j = 1, n_days
        CALL multilevel_matrix(nl, mu(j,isite), rs_true(isite), t(:,isite), &
             w_true, d_true, fup, fdn, fab)
        fup_obs(j,isite) = fup(nlevels_tot)
     END DO
  END DO
  WRITE(*,'(A,I0,A)') 'Generated ', n_sites*n_days, ' synthetic references.'

  !
  ! Initial guess -- deliberately different from truth
  !

  p(1) = 0.5_r_std      ! w initial guess
  p(2) = 0.5_r_std      ! d initial guess
  DO isite = 1, n_sites
     p(2+isite) = 0.5_r_std   ! rs initial guess, same for all sites
  END DO

  DO k = 1, np
     p(k) = MIN(MAX(p(k), p_min(k)), p_max(k))
  END DO

  WRITE(*,'(A)') ''
  WRITE(*,'(A,F10.6,A,F10.6)') 'Initial guess: w = ', p(1), '  d = ', p(2)
  WRITE(*,'(A,F10.6)') 'Initial guess: rs (all sites) = ', p(3)

  !
  ! Set gradient tolerance based on the initial gradient magnitude
  ! (residual-weighted gradient, i.e. actual dJ/dp at the start,
  ! not the raw sensitivity row used inside the main LM loop)
  !

  BLOCK
     REAL(KIND=r_std) :: test_grad(np), r_val
     INTEGER :: test_site

     test_site = 1

     CALL multilevel_matrix(nl, mu(1,test_site), p(2+test_site), t(:,test_site), &
          p(1), p(2), fup, fdn, fab)

     r_val = fup(nlevels_tot) - fup_obs(1,test_site)

     fupb = 0.0_r_std
     fupb(nlevels_tot) = r_val
     fdnb = 0.0_r_std
     fabb = 0.0_r_std
     tb   = 0.0_r_std
     wb   = 0.0_r_std
     db   = 0.0_r_std
     mub  = 0.0_r_std
     rsb  = 0.0_r_std

     CALL multilevel_matrix_b(nl, mu(1,test_site), mub, p(2+test_site), rsb, &
          t(:,test_site), tb, p(1), wb, p(2), db, &
          fup, fupb, fdn, fdnb, fab, fabb)

     test_grad = 0.0_r_std
     test_grad(1) = wb
     test_grad(2) = db
     test_grad(2+test_site) = rsb

     gradient_tolerance = 1.0e-8_r_std * MAXVAL(ABS(test_grad))
     gradient_tolerance = MAX(1.0e-14_r_std, gradient_tolerance)

     WRITE(*,'(A)') ''
     WRITE(*,'(A,ES12.4)') 'Max |gradient| at initial point (site 1) = ', MAXVAL(ABS(test_grad))
     WRITE(*,'(A,ES12.4)') 'Gradient tolerance                       = ', gradient_tolerance
  END BLOCK

  !
  ! Optimization settings
  !

  max_iter      = 200
  converged     = .FALSE.
  lambda_lm     = 1.0e-3_r_std
  lambda_lm_min = 1.0e-12_r_std
  lambda_lm_max = 1.0e12_r_std

  OPEN(unit_history, FILE=history_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_history,'(A)') '# w,d,rs adjoint Levenberg-Marquardt calibration history'
  WRITE(unit_history,'(A)') '# Columns: iter cost grad_norm lambda_lm step_norm max_gradient'

  !
  ! Initial cost
  !

  cost = 0.0_r_std
  DO isite = 1, n_sites
     DO j = 1, n_days
        CALL multilevel_matrix(nl, mu(j,isite), p(2+isite), t(:,isite), p(1), p(2), fup, fdn, fab)
        cost = cost + 0.5_r_std * (fup(nlevels_tot) - fup_obs(j,isite))**2
     END DO
  END DO

  WRITE(*,'(A)') ''
  WRITE(*,'(A,ES18.10)') 'Initial cost = ', cost

  !
  ! OPTIMIZATION LOOP
  !

  DO iter = 1, max_iter

     old_cost = cost
     grad_p = 0.0_r_std
     Hess   = 0.0_r_std
     cost   = 0.0_r_std

     !
     ! Build full J^T J and J^T r via one adjoint call per (site,day).
     ! Seeding fupb = 1.0 (not the residual) gives the RAW sensitivity
     ! row [dFup/dw, dFup/dd, dFup/drs(isite)] -- one Jacobian row.
     !

     DO isite = 1, n_sites
        DO j = 1, n_days

           CALL multilevel_matrix(nl, mu(j,isite), p(2+isite), t(:,isite), &
                p(1), p(2), fup, fdn, fab)

           fupb = 0.0_r_std
           fupb(nlevels_tot) = 1.0_r_std     ! raw sensitivity seed
           fdnb = 0.0_r_std
           fabb = 0.0_r_std
           tb   = 0.0_r_std
           wb   = 0.0_r_std
           db   = 0.0_r_std
           mub  = 0.0_r_std
           rsb  = 0.0_r_std

           CALL multilevel_matrix_b(nl, mu(j,isite), mub, p(2+isite), rsb, &
                t(:,isite), tb, p(1), wb, p(2), db, &
                fup, fupb, fdn, fdnb, fab, fabb)

           row_grad = 0.0_r_std
           row_grad(1)       = wb
           row_grad(2)       = db
           row_grad(2+isite) = rsb

           BLOCK
             REAL(KIND=r_std) :: r_val
             r_val = fup(nlevels_tot) - fup_obs(j,isite)
             cost   = cost   + 0.5_r_std * r_val**2
             grad_p = grad_p + r_val * row_grad          ! J^T r
             DO k = 1, np
                Hess(k,:) = Hess(k,:) + row_grad(k) * row_grad(:)   ! J^T J
             END DO
           END BLOCK

        END DO
     END DO

     grad_norm    = SQRT(SUM(grad_p**2))
     max_gradient = MAXVAL(ABS(grad_p))

     IF (grad_norm < gradient_tolerance) THEN
        WRITE(*,'(A)') 'Converged: gradient tolerance reached.'
        converged = .TRUE.
        WRITE(unit_history,'(I8,5(ES20.10))') iter, cost, grad_norm, lambda_lm, 0.0_r_std, max_gradient
        EXIT
     END IF

     IF (cost < cost_tolerance) THEN
        WRITE(*,'(A)') 'Converged: cost tolerance reached.'
        converged = .TRUE.
        EXIT
     END IF

     !
     ! Damped Gauss-Newton step, backtracking
     ! on lambda_lm until the step actually reduces cost.
     !

     accepted = .FALSE.

     DO ls_iter_lm = 1, max_ls_iter

        Hess_lm = Hess
        DO k = 1, np
           Hess_lm(k,k) = Hess(k,k) + lambda_lm * MAX(Hess(k,k), 1.0e-30_r_std)
        END DO

        CALL solve_linear_system(np, Hess_lm, -grad_p, delta_p, singular)

        IF (singular) THEN
           lambda_lm = MIN(10.0_r_std*lambda_lm, lambda_lm_max)
           CYCLE
        END IF

        DO k = 1, np
           p_trial(k) = MIN(MAX(p(k) + delta_p(k), p_min(k)), p_max(k))
        END DO

        cost_new = 0.0_r_std
        DO isite = 1, n_sites
           DO j = 1, n_days
              CALL multilevel_matrix(nl, mu(j,isite), p_trial(2+isite), t(:,isite), &
                   p_trial(1), p_trial(2), fup, fdn, fab)
              cost_new = cost_new + 0.5_r_std*(fup(nlevels_tot)-fup_obs(j,isite))**2
           END DO
        END DO

        IF (cost_new < cost) THEN
           accepted = .TRUE.
           EXIT
        END IF

        lambda_lm = MIN(10.0_r_std*lambda_lm, lambda_lm_max)

     END DO

     IF (accepted) THEN
        actual_reduction    = cost - cost_new
        predicted_reduction = -SUM(grad_p*delta_p) - 0.5_r_std*DOT_PRODUCT(delta_p, MATMUL(Hess,delta_p))
        IF (predicted_reduction > 1.0e-30_r_std) THEN
           rho = actual_reduction / predicted_reduction
        ELSE
           rho = 0.0_r_std
        END IF

        step_norm = SQRT(SUM((p_trial-p)**2))
        p = p_trial
        cost = cost_new

        IF (rho > 0.75_r_std) THEN
           lambda_lm = MAX(lambda_lm/3.0_r_std, lambda_lm_min)
        ELSE IF (rho < 0.25_r_std) THEN
           lambda_lm = MIN(lambda_lm*4.0_r_std, lambda_lm_max)
        END IF
     ELSE
        WRITE(*,'(A,I6,A)') 'LM step failed at iteration ', iter, '.'
        step_norm = 0.0_r_std
     END IF

     WRITE(unit_history,'(I8,5(ES20.10))') iter, cost, grad_norm, lambda_lm, step_norm, max_gradient
     WRITE(*,'(A,I5,A,ES14.6,A,ES12.4,A,ES10.3,A,ES12.4,A,F10.6,A,F10.6)') &
          'iter=', iter, ' cost=', cost, ' grad=', grad_norm, &
          ' lambda=', lambda_lm, ' step=', step_norm, &
          '  w=', p(1), ' d=', p(2)

     IF (step_norm < parameter_tolerance*(1.0_r_std+SQRT(SUM(p**2)))) THEN
        WRITE(*,'(A)') 'Converged: parameter step tolerance reached.'
        converged = .TRUE.
        EXIT
     END IF

     IF (ABS(old_cost-cost) < cost_tolerance*MAX(1.0_r_std,old_cost)) THEN
        WRITE(*,'(A)') 'Converged: cost change tolerance reached.'
        converged = .TRUE.
        EXIT
     END IF

  END DO

  CLOSE(unit_history)

  !
  ! Diagnostics
  !

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') ''
  WRITE(*,'(A)') ' CALIBRATION COMPLETE'
  WRITE(*,'(A)') ''

  IF (converged) THEN
     WRITE(*,'(A)') 'Status: CONVERGED'
  ELSE
     WRITE(*,'(A)') 'Status: MAXIMUM ITERATIONS REACHED'
  END IF

  WRITE(*,'(A,ES18.10)') 'Final cost = ', cost
  WRITE(*,'(A)') ''
  WRITE(*,'(A,F12.8,A,F12.8,A,ES14.6)') 'w:  true=', w_true, '  recovered=', p(1), '  err=', ABS(p(1)-w_true)
  WRITE(*,'(A,F12.8,A,F12.8,A,ES14.6)') 'd:  true=', d_true, '  recovered=', p(2), '  err=', ABS(p(2)-d_true)
  WRITE(*,'(A)') ''

  DO isite = 1, n_sites
     WRITE(*,'(A,I3,A,F10.6,A,F10.6,A,ES14.6)') &
          'rs(', isite, '): true=', rs_true(isite), &
          '  recovered=', p(2+isite), '  err=', ABS(p(2+isite)-rs_true(isite))
  END DO

  !
  ! Write recovered parameters
  !

  OPEN(unit_profiles, FILE=profile_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_profiles,'(A)') '# Recovered w, d, rs(site)'
  WRITE(unit_profiles,'(A,3ES20.12)') 'w_d : ', p(1), w_true, ABS(p(1)-w_true)
  WRITE(unit_profiles,'(A,3ES20.12)') 'd   : ', p(2), d_true, ABS(p(2)-d_true)
  DO isite = 1, n_sites
     WRITE(unit_profiles,'(A,I0,A,3ES20.12)') 'rs(', isite, ') : ', &
          p(2+isite), rs_true(isite), ABS(p(2+isite)-rs_true(isite))
  END DO
  CLOSE(unit_profiles)

  !
  ! Flux comparison
  !

  OPEN(unit_fluxes, FILE=flux_file, STATUS='REPLACE', ACTION='WRITE')
  WRITE(unit_fluxes,'(A)') '# Final flux comparison'
  WRITE(unit_fluxes,'(A)') '# Columns: site day Fup_model Fup_obs residual'
  DO isite = 1, n_sites
     DO j = 1, n_days
        CALL multilevel_matrix(nl, mu(j,isite), p(2+isite), t(:,isite), p(1), p(2), fup, fdn, fab)
        WRITE(unit_fluxes,'(2I8,3(ES20.12))') isite, j, fup(nlevels_tot), &
             fup_obs(j,isite), fup(nlevels_tot)-fup_obs(j,isite)
     END DO
  END DO
  CLOSE(unit_fluxes)

  WRITE(*,'(A)') ''
  WRITE(*,'(A)') 'True parameters written to: ' // TRIM(true_file)
  WRITE(*,'(A)') 'Optimization history written to: ' // TRIM(history_file)
  WRITE(*,'(A)') 'Recovered parameters written to: ' // TRIM(profile_file)
  WRITE(*,'(A)') 'Flux comparison written to: ' // TRIM(flux_file)

  DEALLOCATE(t, rs_true, mu, mu_offset, mu_amp, fup_obs)
  DEALLOCATE(fup, fdn, fab, fupb, fdnb, fabb, tb)

CONTAINS

  !
  ! Linear systems solver
  !

  SUBROUTINE solve_linear_system(n, A_in, b_in, x, singular)
    INTEGER, INTENT(IN) :: n
    REAL(KIND=r_std), INTENT(IN) :: A_in(n,n), b_in(n)
    REAL(KIND=r_std), INTENT(OUT) :: x(n)
    LOGICAL, INTENT(OUT) :: singular
    REAL(KIND=r_std) :: A(n,n), b(n), factor, pivot
    INTEGER :: i, kpiv, m, maxrow

    A = A_in
    b = b_in
    singular = .FALSE.

    DO kpiv = 1, n-1
       maxrow = kpiv
       DO m = kpiv+1, n
          IF (ABS(A(m,kpiv)) > ABS(A(maxrow,kpiv))) maxrow = m
       END DO
       IF (maxrow /= kpiv) THEN
          CALL swap_rows(A(kpiv,:), A(maxrow,:))
          CALL swap_scalar(b(kpiv), b(maxrow))
       END IF

       pivot = A(kpiv,kpiv)
       IF (ABS(pivot) < 1.0e-30_r_std) THEN
          singular = .TRUE.
          x = 0.0_r_std
          RETURN
       END IF

       DO m = kpiv+1, n
          factor = A(m,kpiv) / pivot
          A(m,kpiv:n) = A(m,kpiv:n) - factor * A(kpiv,kpiv:n)
          b(m) = b(m) - factor * b(kpiv)
       END DO
    END DO

    IF (ABS(A(n,n)) < 1.0e-30_r_std) THEN
       singular = .TRUE.
       x = 0.0_r_std
       RETURN
    END IF

    x(n) = b(n) / A(n,n)
    DO i = n-1, 1, -1
       x(i) = (b(i) - SUM(A(i,i+1:n)*x(i+1:n))) / A(i,i)
    END DO

  END SUBROUTINE solve_linear_system

  SUBROUTINE swap_rows(a, b)
    REAL(KIND=r_std), INTENT(INOUT) :: a(:), b(:)
    REAL(KIND=r_std) :: tmp(SIZE(a))
    tmp = a; a = b; b = tmp
  END SUBROUTINE swap_rows

  SUBROUTINE swap_scalar(a, b)
    REAL(KIND=r_std), INTENT(INOUT) :: a, b
    REAL(KIND=r_std) :: tmp
    tmp = a; a = b; b = tmp
  END SUBROUTINE swap_scalar

END PROGRAM calibrate_w_d_rs
