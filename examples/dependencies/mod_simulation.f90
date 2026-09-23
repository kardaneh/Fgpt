!
!  mod_simulation
!  Orchestrates the run.  Uses grid, math, io, and parameters.
!  This is the first module that touches the whole stack.
!
module mod_simulation
  use mod_parameters
  use mod_grid
  use mod_math
  use mod_io
  implicit none
  public

  real(dp), save :: temperature(n)
  real(dp), save :: albedo(n)
  real(dp), save :: F_olr(n)
  real(dp), save :: F_transport(n)

contains

  !
  ! initialize_state
  !
  subroutine initialize_state()
    implicit none
    integer :: i

    do i = 1, n
      temperature(i) = 288.0_dp - 40.0_dp * sinlat(i)**2
      albedo(i)      = surface_albedo(i, temperature(i))
    end do
  end subroutine initialize_state

  !
  ! compute_olr
  !
  subroutine compute_olr()
    implicit none
    integer :: i
    do i = 1, n
      F_olr(i) = A_olr + B_olr * temperature(i)
    end do
  end subroutine compute_olr

  !
  ! compute_albedo
  !
  subroutine compute_albedo()
    implicit none
    integer :: i
    do i = 1, n
      albedo(i) = surface_albedo(i, temperature(i))
    end do
  end subroutine compute_albedo

  !
  ! step  -- one implicit time step of the EBM
  !
  subroutine step(dt)
    implicit none
    real(dp), intent(in) :: dt

    real(dp) :: lo(n), di(n), up(n), rhs(n), Tnew(n)
    real(dp) :: kface(n), dyface(n)
    real(dp) :: coef
    integer  :: i

    call compute_olr()
    call compute_albedo()

    do i = 1, n
      kface(i)  = D_transport
      dyface(i) = dx_m(i)
    end do

    do i = 1, n
      lo(i)  = 0.0_dp
      di(i)  = 1.0_dp
      up(i)  = 0.0_dp
      rhs(i) = temperature(i)
    end do

    do i = 2, n
      coef    = dt * kface(i) / (C_heat * dyface(i)**2)
      lo(i)   = -coef
      di(i)   = di(i) + coef
      di(i-1) = di(i-1) + coef
      up(i-1) = -coef
    end do

    do i = 1, n
      rhs(i) = rhs(i) + dt * ( &
                 insolation(i) * (1.0_dp - albedo(i)) - F_olr(i) &
               ) / C_heat
    end do

    call solve_tridiag(lo, di, up, rhs, Tnew, n)

    do i = 1, n
      temperature(i) = clamp(Tnew(i), 100.0_dp, 400.0_dp)
    end do
    call compute_olr()
    call compute_transport()
  end subroutine step

  !
  ! compute_transport  -- F = -D dT/dy, positive northward
  !
  subroutine compute_transport()
    implicit none
    integer :: i
    F_transport(1) = 0.0_dp
    do i = 2, n
      F_transport(i) = -D_transport * &
                       (temperature(i) - temperature(i-1)) / dx_m(i)
    end do
    F_transport(n) = 0.0_dp
  end subroutine compute_transport

  !
  ! diagnose  -- T minus its area-weighted meanls
  !
  subroutine diagnose(diag)
    implicit none
    real(dp), intent(out) :: diag(n)
    real(dp) :: mu
    integer  :: i
    mu = weighted_mean(temperature, area, n)
    do i = 1, n
      diag(i) = temperature(i) - mu
    end do
  end subroutine diagnose

  !
  ! energy_budget  -- area-weighted TOA imbalance
  !
  function energy_budget() result(imbalance)
    implicit none
    real(dp) :: imbalance
    real(dp) :: absorbed(n), emitted(n)
    integer  :: i
    do i = 1, n
      absorbed(i) = insolation(i) * (1.0_dp - albedo(i))
      emitted(i)  = F_olr(i)
    end do
    imbalance = sum(area * (absorbed - emitted))
  end function energy_budget

  !
  ! report
  !
  subroutine report(step_number)
    implicit none
    integer, intent(in) :: step_number
    real(dp) :: diag(n)
    real(dp) :: mu, sd, med, wm
    real(dp) :: pole_to_pole, corr_insol

    call diagnose(diag)
    mu   = mean(temperature, n)
    sd   = stddev(temperature, n)
    med  = median(temperature, n)
    wm   = weighted_mean(temperature, area, n)
    pole_to_pole = temperature(1) - temperature(n)
    corr_insol   = correlation(insolation, temperature, n)

    print *, " Step                      : ", step_number
    print *, " Mean T (K)                : ", mu
    print *, " Std dev (K)               : ", sd
    print *, " Median (K)                : ", med
    print *, " Area-weighted mean T (K)  : ", wm
    print *, " Pole-to-pole gradient (K) : ", pole_to_pole
    print *, " corr(insolation, T)       : ", corr_insol
    print *, " TOA imbalance (W/m^2)     : ", energy_budget()
    print *, " Albedo min / max          : ", minval(albedo), maxval(albedo)

    call write_profile_table(gphit, temperature, F_olr, F_transport, n, &
                             "/home/kardaneh/Fgpt/benchmark/demo-ios/dependencies_profile.txt")
    call write_state(temperature, F_olr, F_transport, n, "/home/kardaneh/Fgpt/benchmark/demo-ios/dependencies_state.bin")
  end subroutine report

  !
  ! verify
  !
  subroutine verify()
    implicit none
    real(dp) :: T_in(n), O_in(n), F_in(n)
    integer  :: i
    logical  :: ok

    call read_state(T_in, O_in, F_in, n, "/home/kardaneh/Fgpt/benchmark/demo-ios/dependencies_state.bin")
    ok = .true.
    do i = 1, n
      if (abs(T_in(i) - temperature(i)) > TOL * max(1.0_dp, abs(temperature(i)))) ok = .false.
      if (abs(O_in(i) - F_olr(i))       > TOL * max(1.0_dp, abs(F_olr(i))))       ok = .false.
      if (abs(F_in(i) - F_transport(i)) > TOL * max(1.0_dp, abs(F_transport(i)))) ok = .false.
    end do
    if (ok) then
      print *, " Binary state roundtrip verified."
    else
      print *, " Binary state roundtrip FAILED."
    end if
  end subroutine verify

end module mod_simulation
