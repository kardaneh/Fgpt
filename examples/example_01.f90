
! Subroutine: set_data
!   arr(i) = start + (i-1)*step

subroutine set_data(arr, size, start, step)
    implicit none
    integer, intent(in)  :: size
    real,    intent(out) :: arr(size)
    real,    intent(in)  :: start
    real,    intent(in)  :: step
    integer :: i

    do i = 1, size
        arr(i) = start + real(i - 1) * step
    end do

    print *, "Data set successfully! start=", start, &
                                 " step=", step
end subroutine set_data


! Subroutine: process_data
!   arr(i) = a*arr(i)**2 + b*arr(i) + c

subroutine process_data(arr, size, a, b, c)
    implicit none
    integer, intent(in)    :: size
    real,    intent(inout) :: arr(size)
    real,    intent(in)    :: a, b, c
    integer :: i

    do i = 1, size
        arr(i) = a * arr(i) * arr(i) + b * arr(i) + c
    end do

    print *, "Data processed successfully!", &
                             " a=", a, " b=", b, " c=", c
end subroutine process_data


! Subroutine: display_data
!   Prints the array with a label and summary statistics.

subroutine display_data(arr, size, label)
    implicit none
    integer, intent(in)  :: size
    real,    intent(in)  :: arr(size)
    character(len=*), intent(in) :: label
    real    :: vmin, vmax, vsum
    integer :: i

    print *, trim(label)

    vmin = arr(1)
    vmax = arr(1)
    vsum = 0.0
    do i = 1, size
        print *, " Element ", i, " : ", arr(i)
        if (arr(i) < vmin) vmin = arr(i)
        if (arr(i) > vmax) vmax = arr(i)
        vsum = vsum + arr(i)
    end do

    print *, "   min = ", vmin
    print *, "   max = ", vmax
    print *, "   sum = ", vsum
end subroutine display_data


! Function: compute_average
!   Returns the mean.  Also returns variance and standard deviation
!   through explicit INTENT(OUT) arguments.

function compute_average(arr, size, variance, stddev) result(avg)
    implicit none
    integer, intent(in)  :: size
    real,    intent(in)  :: arr(size)
    real,    intent(out) :: variance
    real,    intent(out) :: stddev
    real :: avg
    real :: var
    integer :: i

    avg = 0.0
    do i = 1, size
        avg = avg + arr(i)
    end do
    avg = avg / real(size)

    var = 0.0
    do i = 1, size
        var = var + (arr(i) - avg)**2
    end do
    var = var / real(size)

    variance = var
    stddev   = sqrt(var)

    print *, "Average computed!"
end function compute_average


! Subroutine: dump_to_file
!   Writes the array to a formatted text file and to an unformatted binary
!   file whose name is derived from the text filename.
!   filename has LEN=15 as in the original.

subroutine dump_to_file(arr, size, filename)
    implicit none
    integer, intent(in) :: size
    real,    intent(in) :: arr(size)
    character(LEN = *), intent(in) :: filename
    integer :: i, unit, ios

    ! ---- Formatted text output ----
    open(newunit=unit, file=trim(filename), status='replace', &
         action='write', iostat=ios)
    if (ios /= 0) then
        print *, "ERROR: could not open ", trim(filename)
        return
    end if
    do i = 1, size
        write(unit, *) arr(i)
    end do
    close(unit)

    print *, "Data written to file: ", trim(filename)

end subroutine dump_to_file


! Subroutine: reset_data
!   Sets arr(i) = value for i in [first, last], and 0.0 elsewhere.

subroutine reset_data(arr, size, value, first, last)
    implicit none
    integer, intent(in)  :: size
    real,    intent(out) :: arr(size)
    real,    intent(in)  :: value
    integer, intent(in)  :: first, last
    integer :: i, lo, hi

    lo = max(1,    first)
    hi = min(size, last)

    do i = 1, size
        arr(i) = 0.0
    end do
    do i = lo, hi
        arr(i) = value
    end do

    print *, "Data reset to ", value, &
          " over [", lo, ",", hi, "]"
end subroutine reset_data


! Subroutine: scale_data
!   arr(i) = arr(i)*factor + offset

subroutine scale_data(arr, size, factor, offset)
    implicit none
    integer, intent(in)    :: size
    real,    intent(inout) :: arr(size)
    real,    intent(in)    :: factor
    real,    intent(in)    :: offset
    integer :: i

    do i = 1, size
        arr(i) = arr(i) * factor + offset
    end do

    print *, "Data scaled by factor: ", factor, &
                                 " offset: ", offset
end subroutine scale_data


! MAIN PROGRAM: test_procedures

program test_procedures
    implicit none

    integer, parameter :: n = 10
    real, allocatable  :: data_array(:)
    real :: average, variance, stddev
    real :: compute_average

    allocate(data_array(n))


    ! Pass 1: simple sequence, default doubling

    call set_data(data_array, n, 0.0, 1.5)
    call display_data(data_array, n, "Initial data display")

    call process_data(data_array, n, 0.0, 2.0, 0.0)   ! arr = 2*arr
    call display_data(data_array, n, "After process_data (2*x)")

    average = compute_average(data_array, n, variance, stddev)
    print *, "Average value: ", average
    print *, "Variance     : ", variance
    print *, "Std deviation: ", stddev

    call scale_data(data_array, n, 0.5, 0.0)
    call display_data(data_array, n, "After scale_data (0.5*x)")


    ! Pass 2: custom sequence, quadratic transform, offset scaling

    call set_data(data_array, n, 1.0, 1.5)
    call display_data(data_array, n, "Pass 2: custom sequence")

    call process_data(data_array, n, 1.0, 0.0, 0.0)   ! arr = arr**2
    call display_data(data_array, n, "Pass 2: squared")

    call scale_data(data_array, n, 2.0, 1.0)          ! arr = 2*arr + 1
    call display_data(data_array, n, "Pass 2: scaled by 2 + 1")


    ! Pass 3: dump, partial reset, verify

    call dump_to_file(data_array, n, "output_data.txt")

    call reset_data(data_array, n, 9.0, 2, 4)
    call display_data(data_array, n, "Pass 3: partial reset to 9.0")

    call dump_to_file(data_array, n, "output_rst.txt")

    call reset_data(data_array, n, 0.1, 3, 5)
    call display_data(data_array, n, "Final: full reset to 0.0")

    deallocate(data_array)

    print *, "Program completed successfully!"
end program test_procedures
