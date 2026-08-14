# -----------------------------
# Environment
# -----------------------------
WORK = /scratchu/kardaneh/tmp
FC_NVIDIA = mpif90
CC        = mpicc

# -----------------------------
# Libraries
# -----------------------------
NETCDF_INCDIR = -I$(NETCDF_C_ROOT)/include -I$(NETCDF_FORTRAN_ROOT)/include
NETCDF_LIBDIR = -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETCDF_FORTRAN_ROOT)/lib -lnetcdff

IOIPSL_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/IOIPSL/inc
IOIPSL_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/IOIPSL/lib -lioipsl

XIOS_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/XIOS/inc
XIOS_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/XIOS/lib -lxios -lstdc++

ORCHIDEE_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/ORCHIDEE/inc
ORCHIDEE_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/ORCHIDEE/lib -lorchidee

# -----------------------------
# Flags
# -----------------------------
FFLAGS = $(IOIPSL_INCDIR) $(XIOS_INCDIR) $(ORCHIDEE_INCDIR) $(NETCDF_INCDIR)
LDFLAGS = $(ORCHIDEE_LIBDIR) $(IOIPSL_LIBDIR) $(XIOS_LIBDIR) $(NETCDF_LIBDIR)

# Fortran flags
FFLAGS_COMMON = -Wall -g -O0 -Kieee -Ktrap=fp -Mbounds -traceback
FFLAGS_CPU    = $(FFLAGS_COMMON) -r8 -i4
FFLAGS_GPU    = $(FFLAGS_COMMON) -r8 -i4 -acc -gpu=cc80

# C flags
CFLAGS_COMMON = -Wall -g -O0 -Kieee -Ktrap=fp -traceback -DUNDERSCORE
CFLAGS_CPU    = $(CFLAGS_COMMON)
CFLAGS_GPU    = $(CFLAGS_COMMON) -acc -gpu=cc80

ifeq ($(strip $(MODE)),GPU)
	FFLAGS_NVIDIA = $(FFLAGS_GPU)
	CFLAGS_NVIDIA = $(CFLAGS_GPU)
else
	FFLAGS_NVIDIA = $(FFLAGS_CPU)
	CFLAGS_NVIDIA = $(CFLAGS_CPU)
endif

# -----------------------------
# Directories and sources
# -----------------------------
OBJ_DIR = obj
MOD_DIR = mod
SRC_DIR = $(SUBDIR_PATH)

# Fortran sources
MODULE_SRC = $(wildcard $(SRC_DIR)/module_*.f90)
MAIN_SRC   = $(wildcard $(SRC_DIR)/main_*.f90)
OTHER_SRC  = $(filter-out $(MODULE_SRC) $(MAIN_SRC), $(wildcard $(SRC_DIR)/*.f90))

MODULE_OBJ = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%,$(MODULE_SRC:.f90=.o))
MAIN_OBJ   = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%,$(MAIN_SRC:.f90=.o))
OTHER_OBJ  = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%,$(OTHER_SRC:.f90=.o))

# Check if Tapenade files exist and need adStack.o
# Detect both adjoint (_b) and tangent (_d) Tapenade-generated files
TAPENADE_ADJOINT_FILES = $(wildcard $(SRC_DIR)/module_*_b.f90)
TAPENADE_TANGENT_FILES = $(wildcard $(SRC_DIR)/module_*_d.f90)
TAPENADE_FILES = $(TAPENADE_ADJOINT_FILES) $(TAPENADE_TANGENT_FILES)

ifneq ($(TAPENADE_FILES),)
  $(info Tapenade files detected (adjoint: $(notdir $(TAPENADE_ADJOINT_FILES)), tangent: $(notdir $(TAPENADE_TANGENT_FILES))) - enabling Tapenade runtime)
  TAPENADE_HOME = /home/kardaneh/tapenade/tapenade_3.16
  TAPENADE_LIB  = $(TAPENADE_HOME)/ADFirstAidKit
  TAPENADE_OBJ = $(OBJ_DIR)/adStack.o
  TAPENADE_SRC = $(TAPENADE_LIB)/adStack.c
  NEEDS_TAPENADE = 1
else
  $(info Automatic differentiation disabled)
  TAPENADE_OBJ =
  TAPENADE_SRC =
  NEEDS_TAPENADE =
endif

ALL_OBJ = $(MODULE_OBJ) $(OTHER_OBJ) $(MAIN_OBJ) $(TAPENADE_OBJ)

# -----------------------------
# Executable and report
# -----------------------------
EXECUTABLE = $(notdir $(CURDIR))
COMPILER_REPORT = $(SRC_DIR)/$(EXECUTABLE).txt

# -----------------------------
# Targets
# -----------------------------
.PHONY: all clean

all: clean $(EXECUTABLE)

$(EXECUTABLE): $(ALL_OBJ)
	@echo "Linking $(EXECUTABLE) ..."
ifdef NEEDS_TAPENADE
	@echo "Including Tapenade object files for automatic differentiation."
endif
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $^ $(LDFLAGS) -o $@ >> $(COMPILER_REPORT) 2>&1
	@echo "Build complete: $(EXECUTABLE)"

# -----------------------------
# Compilation rules
# -----------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR) >> $(COMPILER_REPORT) 2>&1

# Tapenade object - build if needed
ifdef NEEDS_TAPENADE
$(OBJ_DIR)/adStack.o: $(TAPENADE_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS_NVIDIA) -fopenmp -c $< -o $@ >> $(COMPILER_REPORT) 2>&1
endif

# -----------------------------
# Clean
# -----------------------------
clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE) $(MOD_DIR) $(COMPILER_REPORT)
	@echo "Cleaned build artifacts."
