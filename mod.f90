


!! ================================================================================================================================
!! SUBROUTINE   : hydrol_root_profile
!!
!>\BRIEF         Calculates the share of the root biomass in each soil layer based on
!!               structural and functional approach.Calculate the root profile
!!
!! DESCRIPTION  : Root structure is probably how most
!! of us think about roots (i.e. digging a whole and observing where the roots
!! are). When thinking at root structure, the profile should be relatively
!! constant over time. A logic time integrator to determine this constancy is
!! longevity_root as the profile cannot grow faster then the roots grow and die.
!! Currently the structural root profile is simply fixed it over time. This could
!! be changed by, for example, making humcste a function of the tree diameter.

!! In ORCHIDEE root structure is used in the calculation of kfact_root which is water
!! infiltration along roots (accounted for in hydrol.f90) and the input of soil
!! carbon and nitrogen at depth due to the turnover of roots which is accounted
!! for stomate_soil_carbon_discretization.f90. Furthermore, it is used to calculate
!! the root temperature in stomate_resp.f90. When thinking about root function it
!! is not so important where the roots are located but it is more important at
!! which depth the roots will be active. The function approach could be used to
!! calculate from which soil layers the plants take most the soil water for their
!! transpiration. This way of looking at the roots is similar to how we look at
!! the canopy where we have a lot of leaves at places in the canopy where little
!! light can penetrate and where a large part of the photosynthesis is taken care
!! of by the leaves in the top layers of the canopy.
!!
!! NOTE: for the moment root structure and root function are only coupled through
!! the depth of the soil. In the absence of roots, the root functions cannot be
!! fullfilled. This is the most minimalistic coupling. It basically implies that
!! a very small fraction of the roots, e.g., < 1% could take up all the water
!! required for transpiration. A thighter coupling between structure and function
!! is to be expected but this needs to be checked in the literature.
!!
!! RECENT CHANGE(S) : None
!!
!! MAIN OUTPUT VARIABLE(S) : root_profile
!!
!! REFERENCE(S) :
!!
!! \n
!_ ================================================================================================================================

SUBROUTINE hydrol_root_profile_acc(error_flag_hydrol_root_profile_1, error_flag_hydrol_root_profile_2, ji, kjpindex, altmax, sm, smw, root_profile, root_depth)
  !$ACC ROUTINE SEQ

  !! 0.1 Input variables
  INTEGER(KIND = i_std), INTENT(IN) :: kjpindex
  INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_2
  INTEGER(KIND = i_std), INTENT(INOUT) :: error_flag_hydrol_root_profile_1
  INTEGER(KIND = i_std), INTENT(IN) :: ji
  !! Domain size
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm), INTENT(IN) :: altmax
  !! Maximul active layer thickness (m). Be careful, here active means not frozen.
  !! Not related with the active soil carbon pool.
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: sm
  !! Soil moisture of each layer (liquid phase)
  !!  @tex $(kg m^{-2})$ @endtex
  REAL(KIND = r_std), DIMENSION(kjpindex, nslm), INTENT(IN) :: smw
  !! Soil moisture of each layer at wilting point
  !!  @tex $(kg m^{-2})$ @endtex

  !! 0.2 Output variables
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, nslm, nroot_prof), INTENT(OUT) :: root_profile
  !! Normalized root mass/length fraction in each soil layer
  !! (0-1, unitless)
  !! DIM = kjpindex * nvm * nslm
  REAL(KIND = r_std), DIMENSION(kjpindex, nvm, ndepths), INTENT(OUT) :: root_depth
  !! Node and interface numbers at which the deepest roots
  !! occur (1 to nslm, unitless)


  !! 0.3 Modified variables

  !! 0.4 Local variables
  INTEGER(KIND = i_std) :: jv
  INTEGER(KIND = i_std) :: jsl
  !! Indices
  REAL(KIND = r_std) :: rpc
  !! Integration constant for vertical decomposer
  REAL(KIND = r_std) :: z_top
  REAL(KIND = r_std) :: z_bottom
  !! top and bottom node in between which to integrate the root profile
  REAL(KIND = r_std) :: count
  !! Count the number of errors
  REAL(KIND = r_std), DIMENSION(nslm) :: root_profile_tmp
  !! Temporary variable
  REAL(KIND = r_std) :: root_depth_tmp
  REAL(KIND = r_std) :: minmax_value
  INTEGER(KIND = i_std) :: minmax_index
  !! Temporary variable

  !_ ================================================================================================================================

  !! 1.Rooting depth

  ! Calculate the maximum depth roots occur at. Two constraints are already accouned
  ! for: (1) crop can root to 0.8 m, grasslands are assumed to root no deeper than 1 m.
  ! Trees can root down to 2 m. (2) Roots do not extend into frozen soils.
  ! ORCHIDEE assumes that plant roots go down to the depth of the soil water profile
  ! right from day 1. In other words, even a very young tree sapling, crop or grasslands
  ! has roots that extend down to their :: max_root_depth.
  ! Hence, plant age/height does NOT constraint rooting depth for now. A future
  ! development could make rooting depth a function of plant height. This approach
  ! would correctly increase the water stress of young vegetation but may wrongly
  ! increase the water stress of young plants growing in semi-arid regions. In such
  ! regions vegetative reproduction is likley to be common as it allows the offspring
  ! to use water from the parent plant as long as the offsprong is too small to reach
  ! the deeper water layers.
  root_depth(ji, :, :) = zero
  DO jv = 1, nvm

    ! Plants can root up to 2 m (zdr(jsl), the prescribed max_root_depth
    ! or the first permafrost layer
    root_depth_tmp = MIN(MIN(altmax(ji, jv), maxaltmax), MIN(zdr(nslm), max_root_depth(jv)))

    ! Find the index of the node which is the closest to the actual
    ! root_depth. This layer is used to truncate the root profile.
    minmax_value = HUGE(0.0)
    minmax_index = 1
    DO jsl = 1, nslm
      IF (ABS(root_depth_tmp - znh(jsl)) .LT. minmax_value) THEN
        minmax_value = ABS(root_depth_tmp - znh(jsl))
        minmax_index = jsl + 1 - 1
      END IF
    END DO
    root_depth(ji, jv, inode) = minmax_index

    ! zdr is defined as a 0:nslm matrix. MINLOC does not know that
    ! and gives the indices ad 1:nslm+1. Convert the indices back to
    ! 0:nslm by subtracting 1.
    minmax_value = HUGE(0.0)
    minmax_index = 1
    DO jsl = 0, nslm
      IF (ABS(root_depth_tmp - zdr(jsl)) .LT. minmax_value) THEN
        minmax_value = ABS(root_depth_tmp - zdr(jsl))
        minmax_index = jsl + 1 - 0
      END IF
    END DO
    root_depth(ji, jv, iinterface) = minmax_index - 1

  END DO
  ! jv
    DO jv = 1, nvm
    IF (root_depth(ji, jv, inode) .LE. 2) THEN
      root_depth(ji, jv, inode) = 2
    END IF
  END DO
  DO jv = 1, nvm
    IF (root_depth(ji, jv, iinterface) .LE. 2) THEN
      root_depth(ji, jv, iinterface) = 2
    END IF
  END DO


  !! 2.Structural root profile

  ! The structural root profile is calculated as an exponentially decreasing
  ! root mass with depth which. ORCHIDEE uses the structural root profile to
  ! calculate rain infiltration along the roots, som inputs at depth and the
  ! root temperature for autotrophic respiration.

  ! NOTE: the shape of the profile is determined by its depth and the parameter
  ! humcste. For the moment humcste is PFT-dependent but constant over time. The
  ! depth is PFT-dependent (see above) and is a function of the active layer
  ! thickness when the soil discretization is used. The depth of the root profile
  ! could/should be made a function of tree diameter or plant biomass.
  root_profile(ji, :, :, istruc) = zero
  DO jv = 1, nvm

    ! Note that the integration will start at the top of the second layer (the
    ! top of the first layer is zdr(0) hence the zdr(1) is the top of the
    ! second layer) and will continue until the bottom of the profile. The
    ! bottom is the profile is calculated above but never extends deeper than
    ! the depth used in hydrol.f90 (in thermosoil.f90 the profile extends much
    ! deeper. The first layer is excluded because the profile will be used to
    ! extract water from the soil. The first layer is very thin and by extracting
    ! water it could dry to quickly. Also in reality not too many roots are found
    ! in the top mm of the soil. zdr describes the nodes and interfaces of the soil
    ! layers as proposed by de Rosnay 1999 (PhD thesis).
    z_top = zdr(1)
    z_bottom = zdr(root_depth(ji, jv, iinterface))

    ! Calculate the total surface area under an exponential curve between
    ! zdr(1) and the zdr(nslm)
    rpc = un / (EXP(- z_top * humcste(jv)) - EXP(- z_bottom * humcste(jv)))

      DO jsl = 2, root_depth(ji, jv, iinterface)

      ! Calculate the share of the total roots for layers which are "centered"
      ! at the nodes of the hydrology scheme. Centered was written in quotes
      ! because the layer is not symmetric. Using the nodes as the center of the
      ! layers follows De Rosnay (PhD, figure C.2 page 156). The root profile
      ! starts at the top of the second layer, ends at the bottom of the 11th
      ! layer and is calculated for the nodes in between.
      ! The following equation are derived (but rewritten) from the integrals
      ! of equations C9 to C11 of De Rosnay's (1999) PhD thesis (page 158).
      root_profile(ji, jv, jsl, istruc) = rpc * (EXP(- zdr(jsl - 1) * humcste(jv)) - EXP(- zdr(jsl) * humcste(jv)))

    END DO
    ! root_depth

    ! Top layer does not contain structural roots (see z_top)
    ! This line is not needed but was added as a reminder.
    root_profile(ji, jv, 1, istruc) = zero

      ! Error checking. Each root profile should add up to 1.
      IF (err_act .GT. 1) THEN
      IF (ABS(SUM(root_profile(ji, jv, :, istruc)) - un) .GT. 100 * EPSILON(un)) THEN
        error_flag_hydrol_root_profile_1 = error_flag_hydrol_root_profile_1 + 1
      END IF
    END IF

    ! kjpindex
  END DO
  ! nvm


  !! 3. Functional root profile

  ! Calculates the share of the root biomass in each soil layer based on
  ! the soil water content in each layer. The roots now follow the water.
  ! this results in a very dynamic root profile that may change every half
  ! hour (i.e. the time step for hydrol). ORCHIDEE uses the root function
  ! to calculate plant water stress (hydraulic_arch.f90) and the soil layers
  ! from which the transpiration is taken (hydrol.f90).

  ! NOTE: yet anonther root function is nutrient uptake. A separate root
  ! profile could be calculated to be used in the calculation of N uptake
  ! is stomate_soilcarbon.f90 (nitrogen_dynamics).
  root_profile(ji, :, :, ifunc) = zero
  DO jv = 2, nvm

    ! Plant available soil water per layer.
    ! The calculations could make use of sm and smw. These variables has
    ! been labelled soil moisture in kg/m2 and soil moisture at each layer
    ! at wilting point also in kg/m2. As an alternative swc and mcr could
    ! be used. swc is calculated in hydrol.f90 based on mc (m3/m3). mc is
    ! used to calculate smt (total soil water thus liquid + ice) and mcl is
    ! used to calculate sm (liquid only). sm denotes only liquid water, swc
    ! denotes liquid and frozen water. Given the focus on root function, a
    ! variable describing the liquid water seems the better choice.

    !+++CHECK+++
    ! The most important seems to be the difference in the dimensions of the
    ! variables: the dimensions of sm variables are kjpindex,nslm the
    ! dimension of the swc variable is kjpindex,nslm,nst. For the application
    ! we have in mind a different root profile for each soil tile (nst) seems
    ! desirable. Note that the difference in dimensions reflect the spatial
    ! scale of the model but it is not clear at which scale one approach
    ! (tile vs pixel) would be really prefered above another. Should we use
    ! the commented code further below?
    root_profile_tmp(:) = zero
    DO jsl = 2, root_depth(ji, jv, iinterface)
      root_profile_tmp(jsl) = MAX(zero, sm(ji, jsl) - smw(ji, jsl))
    END DO
    !+++++++++++

      ! Normalize to obtain a root profile (fraction between 0 and 1)
      IF (SUM(root_profile_tmp(:)) .GT. min_sechiba) THEN
      root_profile(ji, jv, :, ifunc) = root_profile_tmp(:) / SUM(root_profile_tmp(:))
    ELSE
      root_profile(ji, jv, :, ifunc) = zero
    END IF

    ! Top layer is not used for water uptake
    ! This line is not needed but was added as a reminder.
    root_profile(ji, jv, 1, ifunc) = zero

      ! The functional profile should also equal to one if there is soil moisture.
      IF (err_act .GT. 1) THEN
      IF (ABS(SUM(root_profile(ji, jv, :, ifunc)) - un) .GT. 100 * EPSILON(un) .AND. SUM(root_profile_tmp(:)) .GT. min_sechiba) THEN
        error_flag_hydrol_root_profile_2 = error_flag_hydrol_root_profile_2 + 1
      END IF
    END IF

    ! kjpindex
  END DO
  ! ivm


  !+++CHECK+++
  ! The code above gives one root profile per pixel. The approach below would
  ! enable calculating one root profile per soil tile (bare, short and
  ! tall vegetation). NOTE that the variable name needs to be changed
  ! from root_dens to root_profile and that the dimensions need to re-
  ! ordered.
  !!$    DO ivm = 1, nvm
  !!$
  !!$       ! Link the pft to the soil tile
  !!$       istm = pref_soil_veg(ivm)
  !!$       IF ( is_tree(ivm) ) THEN
  !!$
  !!$          nroot_tmp(:) = zero
  !!$          ! Plant available soil water per layer. mcr is the residual soil
  !!$          ! water and depends on the soil type which is stored in njsc(ipts)
  !!$          DO ibdl = 2, nslm
  !!$             nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
  !!$          ENDDO
  !!$
  !!$       ELSE
  !!$
  !!$          ! Specific case for grasses where we only consider the first 1m of soil.
  !!$          ! Plant available soil water per layer. mcr is the residual soil
  !!$          ! water and depends on the soil type which is stored in njsc(ipts)
  !!$          nroot_tmp(:) = zero
  !!$          DO ibdl = 2, nslm
  !!$             IF (znt(ibdl) .LT. un) THEN
  !!$                nroot_tmp(ibdl) = MAX(zero,swc(ipts,ibdl,istm)-mcr(njsc(ipts)))
  !!$             ELSE
  !!$                nroot_tmp(ibdl) = zero
  !!$             END IF
  !!$          ENDDO
  !!$
  !!$       END IF
  !!$
  !!$       ! Normalize to obtain a root profile (fraction between 0 and 1)
  !!$       IF (SUM(nroot_tmp(:)) .GT. min_sechiba ) THEN
  !!$          root_dens(ipts,:,ivm) = nroot_tmp(:)/SUM(nroot_tmp(:))
  !!$       ELSE
  !!$          root_dens(ipts,:,ivm) = zero
  !!$       END IF
  !!$       root_dens(ipts,1,ivm) = zero
  !!$
  !!$    ENDDO
  !+++++++++++


END SUBROUTINE hydrol_root_profile_acc