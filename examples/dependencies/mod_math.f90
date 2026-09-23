!
!  mod_math
!  Pure statistical and utility functions.  Depends only on mod_parameters.
!
module mod_math
  use mod_parameters
  implicit none
  public

contains

  function mean(x, m) result(mu)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: mu
    integer  :: i
    mu = 0.0_dp
    do i = 1, m
      mu = mu + x(i)
    end do
    mu = mu / real(m, dp)
  end function mean

  function variance(x, m) result(v)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: v, mu
    integer  :: i
    mu = mean(x, m)
    v  = 0.0_dp
    do i = 1, m
      v = v + (x(i) - mu)**2
    end do
    v = v / real(m, dp)
  end function variance

  function stddev(x, m) result(s)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: s
    s = sqrt(variance(x, m))
  end function stddev

  function weighted_mean(x, w, m) result(wm)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m), w(m)
    real(dp) :: wm, wsum
    integer  :: i
    wsum = 0.0_dp
    wm   = 0.0_dp
    do i = 1, m
      wm   = wm   + w(i) * x(i)
      wsum = wsum + w(i)
    end do
    if (wsum > 0.0_dp) then
      wm = wm / wsum
    else
      wm = 0.0_dp
    end if
  end function weighted_mean

  function correlation(x, y, m) result(r)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m), y(m)
    real(dp) :: r, mx, my, sx, sy, sxy
    integer  :: i
    mx = mean(x, m)
    my = mean(y, m)
    sx = 0.0_dp
    sy = 0.0_dp
    sxy = 0.0_dp
    do i = 1, m
      sx  = sx  + (x(i) - mx)**2
      sy  = sy  + (y(i) - my)**2
      sxy = sxy + (x(i) - mx) * (y(i) - my)
    end do
    if (sx > 0.0_dp .and. sy > 0.0_dp) then
      r = sxy / sqrt(sx * sy)
    else
      r = 0.0_dp
    end if
  end function correlation

  function median(x, m) result(med)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    real(dp) :: med, tmp(m), t
    integer  :: i, j

    tmp = x
    do i = 1, m-1
      do j = i+1, m
        if (tmp(j) < tmp(i)) then
          t = tmp(i)
          tmp(i) = tmp(j)
          tmp(j) = t
        end if
      end do
    end do
    if (mod(m, 2) == 1) then
      med = tmp((m+1)/2)
    else
      med = 0.5_dp * (tmp(m/2) + tmp(m/2 + 1))
    end if
  end function median

  function clamp(x, lo, hi) result(y)
    implicit none
    real(dp), intent(in) :: x, lo, hi
    real(dp) :: y
    y = min(max(x, lo), hi)
  end function clamp

  subroutine linspace(x, m, a, b)
    implicit none
    integer,  intent(in)  :: m
    real(dp), intent(in)  :: a, b
    real(dp), intent(out) :: x(m)
    integer  :: i
    if (m == 1) then
      x(1) = a
    else
      do i = 1, m
        x(i) = a + real(i-1, dp) * (b - a) / real(m-1, dp)
      end do
    end if
  end subroutine linspace

  !
  ! solve_tridiag  -- Thomas algorithm
  !
  subroutine solve_tridiag(lo, di, up, d, x, m)
    implicit none
    integer,  intent(in)  :: m
    real(dp), intent(in)  :: lo(m), di(m), up(m), d(m)
    real(dp), intent(out) :: x(m)
    real(dp) :: c(m), dd(m), denom
    integer  :: i

    c(1)  = up(1) / di(1)
    dd(1) = d(1)  / di(1)
    do i = 2, m
      denom = di(i) - lo(i) * c(i-1)
      c(i)  = up(i) / denom
      dd(i) = (d(i) - lo(i) * dd(i-1)) / denom
    end do
    x(m) = dd(m)
    do i = m-1, 1, -1
      x(i) = dd(i) - c(i) * x(i+1)
    end do
  end subroutine solve_tridiag

end module mod_math
