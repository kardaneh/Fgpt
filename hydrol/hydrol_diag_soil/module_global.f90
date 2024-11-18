
MODULE module_global
  IMPLICIT NONE
  INTEGER, PARAMETER :: i_std = 4
  INTEGER, PARAMETER :: r_std = 8
  INTEGER(KIND = i_std), PARAMETER :: nsnow = 3
  INTEGER(KIND = i_std), PARAMETER :: nslm = 11
  INTEGER(KIND = i_std), PARAMETER :: nvm = 15
  INTEGER(KIND = i_std), PARAMETER :: nstm = 3
  INTEGER(KIND = i_std), PARAMETER :: kjpindex = 4717
  INTEGER :: ier
  INTEGER(KIND = i_std) :: ic0, ic
  REAL(KIND = r_std) :: icr, start_time, stop_time
  REAL(KIND = r_std), PARAMETER :: huit = 8._r_std
  INTEGER(KIND = i_std), PARAMETER :: iice = 1
  INTEGER(KIND = i_std), PARAMETER :: imin = 1
  REAL(KIND = r_std), PARAMETER :: min_sechiba = 1.E-8_r_std
  INTEGER(KIND = i_std), PARAMETER :: nnobio = 1
  REAL(KIND = r_std), PARAMETER :: trois = 3._r_std
  REAL(KIND = r_std), PARAMETER :: un = 1._r_std
  REAL(KIND = r_std), PARAMETER :: zero = 0._r_std
  INTEGER :: imax
  LOGICAL :: ok_freeze_cwrr
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ae_ns, ae_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dh
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: dr_ns, dr_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: dz
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: frac_bare_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: humrelv, humrelv_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: humtot, humtot_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: k_lin
  INTEGER(KIND = i_std), ALLOCATABLE, DIMENSION(:, :) :: mask_soiltile
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mc, mc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: mcl
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: profil_froz_hydro, profil_froz_hydro_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: profil_froz_hydro_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: ru_ns, ru_ns_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soil_wet_litter
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soil_wet_ns
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist, soilmoist_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: soilmoist_liquid, soilmoist_liquid_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: soilmoist_s, soilmoist_s_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: subsinksoil
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc, tmc_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_dry_mea, tmc_litt_dry_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_mea, tmc_litt_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: tmc_litt_wet_mea, tmc_litt_wet_mea_cpu
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_adry
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_awet
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_res
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :) :: tmc_litter_sat
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:, :, :) :: vegstressv
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot
  REAL(KIND = r_std), ALLOCATABLE, DIMENSION(:) :: vegtot_old
  !$ACC DECLARE COPYIN(huit, iice, imin, min_sechiba, nnobio, trois, un, zero)
  !$ACC DECLARE CREATE(ae_ns, dh, dr_ns, dz, frac_bare_ns, humrelv, humtot, imax, k_lin, mask_soiltile, mc, mcl, ok_freeze_cwrr,  &
!$ACC& profil_froz_hydro, profil_froz_hydro_ns, ru_ns, soil_wet_litter, soil_wet_ns, soilmoist, soilmoist_liquid, soilmoist_s,  &
!$ACC& subsinksoil, tmc, tmc_litt_dry_mea, tmc_litt_mea, tmc_litt_wet_mea, tmc_litter, tmc_litter_adry, tmc_litter_awet,  &
!$ACC& tmc_litter_res, tmc_litter_sat, vegstressv, vegtot, vegtot_old)
  CONTAINS
  SUBROUTINE declaration_initialization
    OPEN(UNIT = 1363, FILE = '/net/nfs/ssd1/kardaneh/Fgpt/benchmark/hydrol_diag_soil/global.bin', FORM = 'unformatted', STATUS = &
&'old')
    WRITE(*, *) '--- add the declaration and initialization in module global ---'
    READ(1363, IOSTAT = ier) imax
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for imax. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ok_freeze_cwrr
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ok_freeze_cwrr. ', ' IOSTAT : ', ier
    END IF
    IF (.NOT. ALLOCATED(ae_ns)) THEN
      ALLOCATE(ae_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ae_ns_cpu)) THEN
      ALLOCATE(ae_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dh)) THEN
      ALLOCATE(dh(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dr_ns)) THEN
      ALLOCATE(dr_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dr_ns_cpu)) THEN
      ALLOCATE(dr_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(dz)) THEN
      ALLOCATE(dz(nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(frac_bare_ns)) THEN
      ALLOCATE(frac_bare_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humrelv)) THEN
      ALLOCATE(humrelv(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humrelv_cpu)) THEN
      ALLOCATE(humrelv_cpu(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humtot)) THEN
      ALLOCATE(humtot(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(humtot_cpu)) THEN
      ALLOCATE(humtot_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(k_lin)) THEN
      ALLOCATE(k_lin(imin : imax, nslm, kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mask_soiltile)) THEN
      ALLOCATE(mask_soiltile(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc)) THEN
      ALLOCATE(mc(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mc_cpu)) THEN
      ALLOCATE(mc_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(mcl)) THEN
      ALLOCATE(mcl(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro)) THEN
      ALLOCATE(profil_froz_hydro(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_cpu)) THEN
      ALLOCATE(profil_froz_hydro_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(profil_froz_hydro_ns)) THEN
      ALLOCATE(profil_froz_hydro_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ru_ns)) THEN
      ALLOCATE(ru_ns(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(ru_ns_cpu)) THEN
      ALLOCATE(ru_ns_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_litter)) THEN
      ALLOCATE(soil_wet_litter(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soil_wet_ns)) THEN
      ALLOCATE(soil_wet_ns(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist)) THEN
      ALLOCATE(soilmoist(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_cpu)) THEN
      ALLOCATE(soilmoist_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_liquid)) THEN
      ALLOCATE(soilmoist_liquid(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_liquid_cpu)) THEN
      ALLOCATE(soilmoist_liquid_cpu(kjpindex, nslm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_s)) THEN
      ALLOCATE(soilmoist_s(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(soilmoist_s_cpu)) THEN
      ALLOCATE(soilmoist_s_cpu(kjpindex, nslm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(subsinksoil)) THEN
      ALLOCATE(subsinksoil(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc)) THEN
      ALLOCATE(tmc(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_cpu)) THEN
      ALLOCATE(tmc_cpu(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_dry_mea)) THEN
      ALLOCATE(tmc_litt_dry_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_dry_mea_cpu)) THEN
      ALLOCATE(tmc_litt_dry_mea_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_mea)) THEN
      ALLOCATE(tmc_litt_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_mea_cpu)) THEN
      ALLOCATE(tmc_litt_mea_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_wet_mea)) THEN
      ALLOCATE(tmc_litt_wet_mea(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litt_wet_mea_cpu)) THEN
      ALLOCATE(tmc_litt_wet_mea_cpu(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter)) THEN
      ALLOCATE(tmc_litter(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_adry)) THEN
      ALLOCATE(tmc_litter_adry(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_awet)) THEN
      ALLOCATE(tmc_litter_awet(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_res)) THEN
      ALLOCATE(tmc_litter_res(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(tmc_litter_sat)) THEN
      ALLOCATE(tmc_litter_sat(kjpindex, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegstressv)) THEN
      ALLOCATE(vegstressv(kjpindex, nvm, nstm), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegtot)) THEN
      ALLOCATE(vegtot(kjpindex), STAT = ier)
    END IF
    IF (.NOT. ALLOCATED(vegtot_old)) THEN
      ALLOCATE(vegtot_old(kjpindex), STAT = ier)
    END IF
    READ(1363, IOSTAT = ier) ae_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ae_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dh
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dh. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dr_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dr_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) dz
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for dz. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) frac_bare_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for frac_bare_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humrelv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humrelv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) humtot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for humtot. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) k_lin
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for k_lin. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mask_soiltile
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mask_soiltile. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) mcl
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for mcl. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) profil_froz_hydro
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for profil_froz_hydro. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) profil_froz_hydro_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for profil_froz_hydro_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) ru_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for ru_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soil_wet_litter
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soil_wet_litter. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soil_wet_ns
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soil_wet_ns. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist_liquid
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist_liquid. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) soilmoist_s
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for soilmoist_s. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) subsinksoil
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for subsinksoil. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_dry_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_dry_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litt_wet_mea
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litt_wet_mea. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_adry
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_adry. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_awet
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_awet. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_res
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_res. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) tmc_litter_sat
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for tmc_litter_sat. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegstressv
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegstressv. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegtot
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegtot. ', ' IOSTAT : ', ier
    END IF
    READ(1363, IOSTAT = ier) vegtot_old
    IF (ier /= 0) THEN
      WRITE(*, *) 'Error reading from file for vegtot_old. ', ' IOSTAT : ', ier
    END IF
    CLOSE(UNIT = 1363)
    !$ACC UPDATE DEVICE(ae_ns, dh, dr_ns, dz, frac_bare_ns, humrelv, humtot, imax, k_lin, mask_soiltile, mc, mcl, ok_freeze_cwrr,  &
!$ACC& profil_froz_hydro, profil_froz_hydro_ns, ru_ns, soil_wet_litter, soil_wet_ns, soilmoist, soilmoist_liquid, soilmoist_s,  &
!$ACC& subsinksoil, tmc, tmc_litt_dry_mea, tmc_litt_mea, tmc_litt_wet_mea, tmc_litter, tmc_litter_adry, tmc_litter_awet,  &
!$ACC& tmc_litter_res, tmc_litter_sat, vegstressv, vegtot, vegtot_old)
  END SUBROUTINE declaration_initialization
END MODULE module_global
