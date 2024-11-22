
!!
!================================================================================================================================
!! SUBROUTINE   : explicitsnow_gone
!!
!>\BRIEF        Check whether snow is gone
!!
!! DESCRIPTION  : If so, set thickness (and therefore mass and heat) and liquid
!!                content to zero, and adjust fluxes of water, evaporation and
!!                heat into underlying surface.
!! RECENT CHANGE(S) : None
!!
!! MAIN OUTPUT VARIABLE(S): None
!!
!! REFERENCE(S) :
!!
!! FLOWCHART    : None
!! \n
!_
!================================================================================================================================

SUBROUTINE explicitsnow_gone(kjpindex, pgflux, snowheat, snowtemp, snowdz, snowrho, snowliq, grndflux, snowmelt)

  !! 0.1 Input variables
  INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
  !! Domain size
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: pgflux
  !! Net energy into snow pack(w/m2)
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(IN) :: snowheat
  !! Snow heat content (J/m^2)

  !! 0.2 Output variables

  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowtemp
  !! Snow temperature
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowdz
  !! Snow depth [m]
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowrho
  !! Snow density (Kg/m^3)
  REAL(KIND = r_std), DIMENSION(kjpindex, nsnow), INTENT(INOUT) :: snowliq
  !! Liquid water content
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: grndflux
  !! Soil/snow interface heat flux (W/m2)
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(INOUT) :: snowmelt
  !! Snow melt
  REAL(KIND = r_std), DIMENSION(kjpindex) :: thrufal
  !! Water leaving snowpack(kg/m2/s)

  !! 0.4 Local variables

  INTEGER(KIND = i_std) :: jj
  INTEGER(KIND = i_std) :: ji
  REAL(KIND = r_std), DIMENSION(kjpindex) :: snowgone_delta
  REAL(KIND = r_std), DIMENSION(kjpindex) :: totsnowheat
  !!snow heat content at each layer
  REAL(KIND = r_std), DIMENSION(kjpindex) :: snowdepth_crit

  ! first caculate total snowpack snow heat content
  !snowgone_delta(:) = un
  !thrufal(:)=0.0
  !snowmelt(:)=0
  totsnowheat(:) = SUM(snowheat(:, :), DIM = 2)


  !DO ji = 1, kjpindex

  !   IF ( pgflux(ji) >= (-totsnowheat(ji)/dt_sechiba) ) THEN
  !      ! all the snow melts
  !      grndflux(ji) = pgflux(ji) + (totsnowheat(ji)/dt_sechiba)
  !      thrufal(ji)=SUM(snowrho(ji,:)*snowdz(ji,:))
  !      snowgone_delta(ji) = 0.0
  !      snowmelt(ji) = snowmelt(ji)+thrufal(ji)
  !   ENDIF

  ! update of snow state (either still present or not)
  !   DO jj=1,nsnow
  !      snowdz(ji,jj)  =   snowdz(ji,jj) *snowgone_delta(ji)
  !      snowliq(ji,jj)   =   snowliq(ji,jj) *snowgone_delta(ji)
  !      snowtemp(ji,jj) = (1.0-snowgone_delta(ji))*tp_00 + snowtemp(ji,jj)*snowgone_delta(ji)
  !   ENDDO
  !ENDDO

END SUBROUTINE explicitsnow_gone