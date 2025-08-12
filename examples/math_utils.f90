module math_utils
  implicit none
  integer :: global_counter = 0
  real :: pi = 3.14159

contains

  subroutine increment_counter(value)
    integer, intent(in) :: value
    global_counter = global_counter + value
  end subroutine increment_counter

end module math_utils

