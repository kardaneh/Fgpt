
!! ================================================================================================================================
!! SUBROUTINE   : hydrol_soil_froz
!!
!>\BRIEF        Computes profil_froz_hydro_ns, the fraction of frozen water in the soil layers.
!!
!! DESCRIPTION  :
!!
!! RECENT CHANGE(S) : Created by A. Ducharne in 2016.
!!
!! MAIN OUTPUT VARIABLE(S) : profil_froz_hydro_ns
!!
!! REFERENCE(S) :
!!
!! FLOWCHART    : None
!! \n
!_ ================================================================================================================================
!_ hydrol_soil_froz

SUBROUTINE hydrol_soil_froz(nvan, avan, mcr, mcs, kjpindex, ins, njsc, stempdiag)

  IMPLICIT NONE
  !
  !! 0. Variable and parameter declaration

  !! 0.1 Input variables
  INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
  !! Domain size
  INTEGER(KIND = i_std), INTENT(IN) :: ins
  !! Index of soil type
  INTEGER(KIND = i_std), DIMENSION(kjpindex), INTENT(IN) :: njsc
  !! Index of the dominant soil textural class in the grid cell (1-nscm, unitless)
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: nvan
  !! Van Genuchten coeficients n (unitless)
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: avan
  !! Van Genuchten coeficients a (mm-1})
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcr
  !! Residual volumetric water content (m^{3} m^{-3})
  REAL(KIND = r_std), DIMENSION(kjpindex), INTENT(IN) :: mcs
  !! Saturated volumetric water content (m^{3} m^{-3})
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: stempdiag
  !! Diagnostic temp profile from thermosoil

  !! 0.2 Output variables

  !! 0.3 Modified variables

  !! 0.4 Local variables

  INTEGER(KIND = i_std) :: jsl, ji, i
  REAL(KIND = r_std) :: x, m
  REAL(KIND = r_std) :: denom
  REAL(KIND = r_std), DIMENSION(kjpindex) :: froz_frac_moy
  REAL(KIND = r_std), DIMENSION(kjpindex) :: smtot_moy
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm) :: mc_ns

  !_ 
!& ================================================================================================================================

  !    ONLY FOR THE (ok_freeze_cwrr) CASE

  ! Calculation of liquid and frozen saturation degrees above residual moisture
  !   x=liquid saturation degree/residual=(mcl-mcr)/(mcs-mcr)
  !   1-x=frozen saturation degree/residual=(mcfc-mcr)/(mcs-mcr) (=profil_froz_hydro)
  ! It's important for the good work of the water diffusion scheme (tridiag) that the total
  ! liquid water also includes mcr, so mcl > 0 even when x=0

  DO jsl = 1, nslm
    DO ji = 1, kjpindex
      ! Van Genuchten parameter for thermodynamical calculation
      m = 1. - 1. / nvan(ji)

        IF ((.NOT. ok_thermodynamical_freezing) .OR. (mc(ji, jsl, ins) .LT. (mcr(ji) + min_sechiba))) THEN
        ! Linear soil freezing or soil moisture below residual
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          x = (stempdiag(ji, jsl) - (fr_center - fr_dT / 2.)) / fr_dT
        ELSE
          x = 0._r_std
        END IF
      ELSE IF (ok_thermodynamical_freezing) THEN
        ! Thermodynamical soil freezing
          IF (stempdiag(ji, jsl) .GE. (fr_center + fr_dT / 2.)) THEN
          x = 1._r_std
        ELSE IF ((stempdiag(ji, jsl) .GE. (fr_center - fr_dT / 2.)) .AND. (stempdiag(ji, jsl) .LT. (fr_center + fr_dT / 2.))) THEN
          ! Factor 2.2 from the PhD of Isabelle Gouttevin
          x = MIN(((mcs(ji) - mcr(ji)) * ((2.2 * 1000. * avan(ji) * (fr_center + fr_dT / 2. - stempdiag(ji, jsl)) * lhf / &
&ZeroCelsius / 10.) ** nvan(ji) + 1.) ** (- m)) / (mc(ji, jsl, ins) - mcr(ji)), 1._r_std)
        ELSE
          x = 0._r_std
        END IF
      END IF

      profil_froz_hydro_ns(ji, jsl, ins) = 1._r_std - x

      mc_ns(ji, jsl) = mc(ji, jsl, ins) / mcs(ji)

    END DO
    ! loop on grid
  END DO

  ! Applay correction on the frozen fraction
  ! Depends on two external parameters: froz_frac_corr and smtot_corr
  froz_frac_moy(:) = zero
  denom = zero
  DO jsl = 1, nslm
    froz_frac_moy(:) = froz_frac_moy(:) + dh(jsl) * profil_froz_hydro_ns(:, jsl, ins)
    denom = denom + dh(jsl)
  END DO
  froz_frac_moy(:) = froz_frac_moy(:) / denom

  smtot_moy(:) = zero
  denom = zero
  DO jsl = 1, nslm - 1
    smtot_moy(:) = smtot_moy(:) + dh(jsl) * mc_ns(:, jsl)
    denom = denom + dh(jsl)
  END DO
  smtot_moy(:) = smtot_moy(:) / denom

    DO jsl = 1, nslm
    profil_froz_hydro_ns(:, jsl, ins) = MIN(profil_froz_hydro_ns(:, jsl, ins) * (froz_frac_moy(:) ** froz_frac_corr) * &
&(smtot_moy(:) ** smtot_corr), max_froz_hydro)
  END DO

END SUBROUTINE hydrol_soil_froz