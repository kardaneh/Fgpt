module a
implicit none
private
public i, f

integer :: i = 5

contains

integer function f(x) result(r)
integer, intent(in) :: x
r = x + 5
end function

integer function g(x) result(r)
integer, intent(in) :: x
r = x - 5
end function

end module
