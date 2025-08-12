module main_module
  use math_utils
  implicit none
  real :: area

contains

  subroutine compute_area(radius)
    real, intent(in) :: radius
    real :: r_squared
    r_squared = radius * radius
    area = pi * r_squared
    call increment_counter(1)
  end subroutine compute_area

end module main_module
