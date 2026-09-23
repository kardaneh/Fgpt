!
!  mod_grid
!  Grid coordinates.  Depends on mod_parameters.
!
module mod_grid
  use mod_parameters
  implicit none
  public

  real(dp), save :: gphit(n)
  real(dp), save :: sinlat(n)
  real(dp), save :: area(n)
  real(dp), save :: dx_m(n)
  real(dp), save :: insolation(n)
  logical,  save :: is_land(n)

contains

  !
  ! fill_grid
  !
  subroutine fill_grid()
    implicit none
    integer  :: i
    real(dp) :: dlat
    real(dp) :: lat_s, lat_n

    dlat = 180.0_dp / real(n, dp)

    do i = 1, n
      gphit(i)  = -90.0_dp + (real(i, dp) - 0.5_dp) * dlat
      sinlat(i) = sin(gphit(i) * DEG2RAD)
    end do

    do i = 1, n
      lat_s   = gphit(i) - 0.5_dp * dlat
      lat_n   = gphit(i) + 0.5_dp * dlat
      area(i) = 0.5_dp * (sin(lat_n * DEG2RAD) - sin(lat_s * DEG2RAD))
    end do
    area = area / sum(area)

    do i = 2, n
      dx_m(i) = EARTH_RADIUS_M * (gphit(i) - gphit(i-1)) * DEG2RAD
    end do
    dx_m(1) = dx_m(2)
    dx_m(n) = dx_m(n-1)

    do i = 1, n
      insolation(i) = (S0 / 4.0_dp) * &
                      (1.0_dp - 0.48_dp * (3.0_dp * sinlat(i)**2 - 1.0_dp) / 2.0_dp)
    end do

    do i = 1, n
      is_land(i) = (abs(gphit(i)) > 25.0_dp .and. abs(gphit(i)) < 55.0_dp)
    end do
  end subroutine fill_grid

  !
  ! grid_info
  !
  subroutine grid_info()
    implicit none
    print *, " Grid size              : ", n
    print *, " Lat min / max          : ", minval(gphit), maxval(gphit)
    print *, " Sum of area weights    : ", sum(area)
    print *, " Mean insolation (W/m^2): ", sum(area * insolation)
  end subroutine grid_info

  !
  ! surface_albedo
  !
  function surface_albedo(i, T) result(alpha)
    implicit none
    integer,  intent(in) :: i
    real(dp), intent(in) :: T
    real(dp) :: alpha

    if (is_land(i)) then
      alpha = albedo_land
    else if (T < T_freeze) then
      alpha = albedo_ice
    else
      alpha = albedo_ocean
    end if
  end function surface_albedo

end module mod_grid
