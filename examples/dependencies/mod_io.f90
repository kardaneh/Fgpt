!
!  mod_io
!  Text and binary I/O helpers.  Depends only on mod_parameters.
!
module mod_io
  use mod_parameters
  implicit none
  public

contains

  subroutine write_text(x, m, filename)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    character(len=*), intent(in) :: filename
    integer :: unit, i
    open(newunit=unit, file=filename, status='replace')
    do i = 1, m
      write(unit, *) x(i)
    end do
    close(unit)
  end subroutine write_text

  subroutine read_text(x, m, filename)
    implicit none
    integer,  intent(in)  :: m
    real(dp), intent(out) :: x(m)
    character(len=*), intent(in) :: filename
    integer :: unit, i, ios
    open(newunit=unit, file=filename, status='old', iostat=ios)
    if (ios /= 0) stop "read_text: cannot open file"
    do i = 1, m
      read(unit, *, iostat=ios) x(i)
      if (ios /= 0) stop "read_text: read error"
    end do
    close(unit)
  end subroutine read_text

  subroutine write_binary(x, m, filename)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: x(m)
    character(len=*), intent(in) :: filename
    integer :: unit
    open(newunit=unit, file=filename, status='replace', form='unformatted')
    write(unit) m
    write(unit) x
    close(unit)
  end subroutine write_binary

  subroutine read_binary(x, m, filename)
    implicit none
    integer,  intent(in)  :: m
    real(dp), intent(out) :: x(m)
    character(len=*), intent(in) :: filename
    integer :: unit, mread, ios
    open(newunit=unit, file=filename, status='old', &
         form='unformatted', iostat=ios)
    if (ios /= 0) stop "read_binary: cannot open file"
    read(unit) mread
    if (mread /= m) stop "read_binary: size mismatch"
    read(unit) x
    close(unit)
  end subroutine read_binary

  !
  ! write_profile_table
  !   One latitude band per row, list-directed.
  !
  subroutine write_profile_table(lat, T, F_olr, F_tr, m, filename)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: lat(m), T(m), F_olr(m), F_tr(m)
    character(len=*), intent(in) :: filename
    integer :: unit, i
    open(newunit=unit, file=filename, status='replace')
    write(unit, *) "# lat  T  OLR  transport"
    do i = 1, m
      write(unit, *) lat(i), T(i), F_olr(i), F_tr(i)
    end do
    close(unit)
  end subroutine write_profile_table

  !
  ! write_state  -- several arrays in one binary file
  !
  subroutine write_state(T, F_olr, F_tr, m, filename)
    implicit none
    integer,  intent(in) :: m
    real(dp), intent(in) :: T(m), F_olr(m), F_tr(m)
    character(len=*), intent(in) :: filename
    integer :: unit
    open(newunit=unit, file=filename, status='replace', form='unformatted')
    write(unit) m
    write(unit) T
    write(unit) F_olr
    write(unit) F_tr
    close(unit)
  end subroutine write_state

  subroutine read_state(T, F_olr, F_tr, m, filename)
    implicit none
    integer,  intent(in)  :: m
    real(dp), intent(out) :: T(m), F_olr(m), F_tr(m)
    character(len=*), intent(in) :: filename
    integer :: unit, mread, ios
    open(newunit=unit, file=filename, status='old', &
         form='unformatted', iostat=ios)
    if (ios /= 0) stop "read_state: cannot open file"
    read(unit) mread
    if (mread /= m) stop "read_state: size mismatch"
    read(unit) T
    read(unit) F_olr
    read(unit) F_tr
    close(unit)
  end subroutine read_state

  subroutine print_vector(label, x, m)
    implicit none
    character(len=*), intent(in) :: label
    integer,          intent(in) :: m
    real(dp),         intent(in) :: x(m)
    integer :: i
    print *, trim(label)
    do i = 1, m
      print *, "   [", i, "] ", x(i)
    end do
  end subroutine print_vector

end module mod_io
