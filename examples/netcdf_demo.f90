
!  netcdf_demo.F90
!  A self-contained example that reads and writes NetCDF files.


program netcdf_demo
  implicit none

  ! ---- Grid dimensions ----
  integer, parameter :: jpi = 8
  integer, parameter :: jpj = 6
  integer, parameter :: jpk = 5

  ! ---- NetCDF IDs and status ----
  integer :: ncid_in, ncid_out
  integer :: xid, yid, zid, tid
  integer :: id_lon, id_lat, id_dep, id_tmask, id_kbot, id_time
  integer :: status
  integer, dimension(4) :: dims4
  integer, dimension(4) :: start4, count4

  ! ---- Data arrays ----
  real, dimension(jpi,jpj)      :: glamf, gphif, glamt, gphit
  real, dimension(jpk)          :: gdept
  real, dimension(jpi,jpj,jpk)  :: tmask
  real, dimension(jpi,jpj)      :: kbot
  real, dimension(1)            :: time_val

  ! ---- Loop counters ----
  integer :: ji, jj, jk

  include "netcdf.inc"


  ! 1 - Build the grid and a synthetic mask

  do jj = 1, jpj
    do ji = 1, jpi
      glamt(ji,jj) = -180.0 + real(ji-1) * (360.0 / real(jpi))
      gphit(ji,jj) =  -90.0 + real(jj-1) * (180.0 / real(jpj))
      glamf(ji,jj) = glamt(ji,jj) - 0.5 * (360.0 / real(jpi))
      gphif(ji,jj) = gphit(ji,jj) - 0.5 * (180.0 / real(jpj))
    end do
  end do

  do jk = 1, jpk
    gdept(jk) = 10.0 * real(jk) ** 1.5
  end do

  ! Ocean where |lat| < 60, land elsewhere; ocean depth shrinks with |lat|
  do jk = 1, jpk
    do jj = 1, jpj
      do ji = 1, jpi
        if (abs(gphit(ji,jj)) < 60.0 .and. jk <= jpk - abs(int(gphit(ji,jj)/20.0))) then
          tmask(ji,jj,jk) = 1.0
        else
          tmask(ji,jj,jk) = 0.0
        end if
      end do
    end do
  end do


  ! 2 - Write the input file

  print '(A)', "--- Writing sample_in.nc ---"

  status = NF_CREATE('sample_in.nc', NF_CLOBBER, ncid_out)
  call hdlerr(status)

  status = NF_DEF_DIM(ncid_out, 'x', jpi, xid)
  call hdlerr(status)
  status = NF_DEF_DIM(ncid_out, 'y', jpj, yid)
  call hdlerr(status)
  status = NF_DEF_DIM(ncid_out, 'z', jpk, zid)
  call hdlerr(status)
  status = NF_DEF_DIM(ncid_out, 'time_counter', NF_UNLIMITED, tid)
  call hdlerr(status)

  dims4(1) = xid
  dims4(2) = yid
  dims4(3) = zid
  dims4(4) = tid

  status = NF_DEF_VAR(ncid_out, 'nav_lon', NF_FLOAT, 2, dims4(1:2), id_lon)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'nav_lat', NF_FLOAT, 2, dims4(1:2), id_lat)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'deptht',  NF_FLOAT, 1, dims4(3),   id_dep)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'tmask',   NF_FLOAT, 4, dims4,      id_tmask)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'time_counter', NF_FLOAT, 1, dims4(4), id_time)
  call hdlerr(status)

  status = NF_PUT_ATT_TEXT(ncid_out, id_time, 'units', 33, &
                           'seconds since 1900-01-01 00:00:00')
  call hdlerr(status)

  status = NF_ENDDEF(ncid_out)
  call hdlerr(status)

  status = NF_PUT_VAR_REAL(ncid_out, id_lon, glamt)
  call hdlerr(status)
  status = NF_PUT_VAR_REAL(ncid_out, id_lat, gphit)
  call hdlerr(status)
  status = NF_PUT_VAR_REAL(ncid_out, id_dep, gdept)
  call hdlerr(status)

  start4(1) = 1
  start4(2) = 1
  start4(3) = 1
  start4(4) = 1
  count4(1) = jpi
  count4(2) = jpj
  count4(3) = jpk
  count4(4) = 1
  time_val(1) = 1.0

  status = NF_PUT_VARA_REAL(ncid_out, id_time, start4(4), count4(4), time_val)
  call hdlerr(status)
  status = NF_PUT_VARA_REAL(ncid_out, id_tmask, start4, count4, tmask)
  call hdlerr(status)

  status = NF_CLOSE(ncid_out)
  call hdlerr(status)

  print '(A)', "--- sample_in.nc written ---"


  ! 3 - Read it back

  print '(A)', "--- Reading sample_in.nc ---"

  status = NF_OPEN('sample_in.nc', NF_NOWRITE, ncid_in)
  if (status /= NF_NOERR) then
    print '(A)', NF_STRERROR(status)
    stop 'Could not open sample_in.nc'
  end if

  status = NF_INQ_VARID(ncid_in, 'nav_lon', id_lon)
  call hdlerr(status)
  status = NF_INQ_VARID(ncid_in, 'nav_lat', id_lat)
  call hdlerr(status)
  status = NF_INQ_VARID(ncid_in, 'deptht',  id_dep)
  call hdlerr(status)
  status = NF_INQ_VARID(ncid_in, 'tmask',   id_tmask)
  call hdlerr(status)

  status = NF_GET_VAR_REAL(ncid_in, id_lon, glamt)
  call hdlerr(status)
  status = NF_GET_VAR_REAL(ncid_in, id_lat, gphit)
  call hdlerr(status)
  status = NF_GET_VAR_REAL(ncid_in, id_dep, gdept)
  call hdlerr(status)

  start4(1) = 1
  start4(2) = 1
  start4(3) = 1
  start4(4) = 1
  count4(1) = jpi
  count4(2) = jpj
  count4(3) = jpk
  count4(4) = 1
  status = NF_GET_VARA_REAL(ncid_in, id_tmask, start4, count4, tmask)
  call hdlerr(status)

  status = NF_CLOSE(ncid_in)
  call hdlerr(status)

  print '(A,2(1x,F10.4))', " glamf max/min: ", maxval(glamt), minval(glamt)
  print '(A,2(1x,F10.4))', " gphit max/min: ", maxval(gphit), minval(gphit)
  print '(A,2(1x,F10.4))', " tmask max/min: ", maxval(tmask), minval(tmask)


  ! 4 - Compute the bottom level kbot(ji,jj)

  kbot(:,:) = 0.0
  do jj = 1, jpj
    do ji = 1, jpi
      if (tmask(ji,jj,1) == 1.0) then
        jk = 2
        do while (jk <= jpk .and. tmask(ji,jj,jk) == 1.0)
          jk = jk + 1
        end do
        kbot(ji,jj) = real(jk - 1)
      end if
    end do
  end do

  print '(A,F6.1)', " Bottom index maximum: ", maxval(kbot)


  ! 5 - Write the output file

  print '(A)', "--- Writing sample_out.nc ---"

  status = NF_CREATE('sample_out.nc', NF_CLOBBER, ncid_out)
  call hdlerr(status)

  status = NF_DEF_DIM(ncid_out, 'x', jpi, xid)
  call hdlerr(status)
  status = NF_DEF_DIM(ncid_out, 'y', jpj, yid)
  call hdlerr(status)
  status = NF_DEF_DIM(ncid_out, 'time_counter', NF_UNLIMITED, tid)
  call hdlerr(status)

  dims4(1) = xid
  dims4(2) = yid
  dims4(3) = tid

  status = NF_DEF_VAR(ncid_out, 'nav_lon', NF_FLOAT, 2, dims4(1:2), id_lon)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'nav_lat', NF_FLOAT, 2, dims4(1:2), id_lat)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'kbot',    NF_FLOAT, 3, dims4,      id_kbot)
  call hdlerr(status)
  status = NF_DEF_VAR(ncid_out, 'time_counter', NF_FLOAT, 1, dims4(3), id_time)
  call hdlerr(status)

  status = NF_PUT_ATT_TEXT(ncid_out, id_time, 'units', 33, &
                           'seconds since 1900-01-01 00:00:00')
  call hdlerr(status)

  status = NF_ENDDEF(ncid_out)
  call hdlerr(status)

  status = NF_PUT_VAR_REAL(ncid_out, id_lon, glamt)
  call hdlerr(status)
  status = NF_PUT_VAR_REAL(ncid_out, id_lat, gphit)
  call hdlerr(status)

  start4(1) = 1
  start4(2) = 1
  start4(3) = 1
  count4(1) = jpi
  count4(2) = jpj
  count4(3) = 1

  status = NF_PUT_VARA_REAL(ncid_out, id_time, start4(3), count4(3), time_val)
  call hdlerr(status)
  status = NF_PUT_VARA_REAL(ncid_out, id_kbot, start4(1:3), count4(1:3), kbot)
  call hdlerr(status)

  status = NF_CLOSE(ncid_out)
  call hdlerr(status)

  print '(A)', "--- sample_out.nc written ---"
  print '(A)', "Program completed successfully!"

contains


  ! Subroutine: hdlerr

  subroutine hdlerr(istatus)
    implicit none
    integer, intent(in) :: istatus
    include 'netcdf.inc'
    if (istatus /= NF_NOERR) then
      print '(A)', NF_STRERROR(istatus)
      stop 'stopped here'
    end if
  end subroutine hdlerr

end program netcdf_demo
