!
!  mod_parameters
!  Global kinds, constants, and sizes.  No dependencies.
!
module mod_parameters
  implicit none
  public

  integer, parameter :: dp = kind(1.0d0)

  ! ---- Grid ----
  integer, parameter :: n = 32                       ! latitude bands
  real(dp), parameter :: PI     = 3.14159265358979323846_dp
  real(dp), parameter :: DEG2RAD = PI / 180.0_dp
  real(dp), parameter :: EARTH_RADIUS_M = 6371000.0_dp

  ! ---- Radiation ----
  real(dp), parameter :: S0       = 1361.0_dp        ! solar constant (W/m^2)
  real(dp), parameter :: sigma    = 5.670374419e-8_dp! Stefan-Boltzmann
  real(dp), parameter :: A_olr    = 210.0_dp         ! OLR offset (W/m^2)
  real(dp), parameter :: B_olr    = 2.0_dp           ! OLR slope (W/m^2/K)

  ! ---- Surface / albedo ----
  real(dp), parameter :: T_freeze         = 273.15_dp
  real(dp), parameter :: albedo_ocean     = 0.10_dp
  real(dp), parameter :: albedo_ice       = 0.62_dp
  real(dp), parameter :: albedo_land      = 0.30_dp

  ! ---- Thermodynamics ----
  real(dp), parameter :: C_heat    = 2.0e7_dp        ! heat capacity (J/m^2/K)
  real(dp), parameter :: D_transport = 0.6_dp        ! diffusion coefficient (W/m^2/K)

  ! ---- Numerics ----
  real(dp), parameter :: TOL       = 1.0e-12_dp
  real(dp), parameter :: dt_max    = 86400.0_dp * 30.0_dp   ! 30 days in seconds

end module mod_parameters
