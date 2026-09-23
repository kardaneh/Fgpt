!
!  main_program
!  Drives the EBM run.
!
program main_program
  use mod_parameters
  use mod_math
  use mod_grid
  use mod_simulation
  implicit none

  real(dp), parameter :: dt = dt_max
  integer,  parameter :: nsteps = 200
  integer,  parameter :: nreport = 25
  real(dp) :: x(n)
  real(dp) :: mu, var, sd, med, r
  integer  :: k

  print *, ""
  print *, " 1-D Energy Balance Model"
  print *, ""

  call fill_grid()
  call grid_info()

  call linspace(x, n, 1.0_dp, 2.0_dp)
  mu  = mean(x, n)
  var = variance(x, n)
  sd  = stddev(x, n)
  med = median(x, n)
  r   = correlation(x, x, n)
  print *, " linspace mean      : ", mu
  print *, " linspace variance  : ", var
  print *, " linspace stddev    : ", sd
  print *, " linspace median    : ", med
  print *, " corr(x, x)         : ", r
  print *, ""

  call initialize_state()

  do k = 1, nsteps
    call step(dt)
    if (mod(k, nreport) == 0) then
      print *, ""
      call report(k)
    end if
  end do

  print *, ""
  call verify()
  print *, "Program completed successfully."

end program main_program
