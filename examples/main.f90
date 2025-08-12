program main
  use main_module
  implicit none
  real :: r

  r = 2.0
  call compute_area(r)
  print *, "Radius: ", r
  print *, "Area: ", area
  print *, "Global counter: ", global_counter

end program main
