# Copyright 2026 IPSL / CNRS / Sorbonne University
# Authors: Shivamshan Sivanesan and Kazem Ardaneh
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/

# -----------------------------
# Environment
# -----------------------------
FC_NVIDIA = mpif90
CC        = mpicc

# -----------------------------
# Flags
# -----------------------------
# Fortran flags (CPU only)
FFLAGS_COMMON = -Wall -g -O0 -Kieee -Ktrap=fp -Mbounds -traceback
FFLAGS_CPU    = $(FFLAGS_COMMON) -r8 -i4
FFLAGS        = $(FFLAGS_CPU)

# C flags (CPU only)
CFLAGS_COMMON = -Wall -g -O0 -Kieee -Ktrap=fp -traceback -DUNDERSCORE
CFLAGS_CPU    = $(CFLAGS_COMMON)
CFLAGS        = $(CFLAGS_CPU)

# -----------------------------
# Directories and sources
# -----------------------------
OBJ_DIR = obj
MOD_DIR = mod
SRC_DIR = .

# Fortran sources
MODULE_SRC = $(wildcard $(SRC_DIR)/module_*.f90)
MAIN_SRC   = $(wildcard $(SRC_DIR)/calibrate.f90)
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
# Executable
# -----------------------------
EXECUTABLE = calibrate

# -----------------------------
# Targets
# -----------------------------
.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(ALL_OBJ)
	@echo "Linking $(EXECUTABLE) ..."
ifdef NEEDS_TAPENADE
	@echo "Including Tapenade object files for automatic differentiation."
endif
	$(FC_NVIDIA) $(FFLAGS) $^ -o $@
	@echo "Build complete: $(EXECUTABLE)"

# -----------------------------
# Compilation rules
# -----------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

# Tapenade object - build if needed
ifdef NEEDS_TAPENADE
$(OBJ_DIR)/adStack.o: $(TAPENADE_SRC)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@
endif

# -----------------------------
# Clean
# -----------------------------
clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE) $(MOD_DIR)
	@echo "Cleaned build artifacts."

# -----------------------------
# Run
# -----------------------------
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# -----------------------------
# Run and visualize
# -----------------------------
viz: $(EXECUTABLE)
	./$(EXECUTABLE)
	@echo "Generating visualization plots..."
	python3 plots.py || python plots.py
	@echo "Visualization complete!"

# -----------------------------
# Just visualize (if data already exists)
# -----------------------------
plot:
	python3 plot_results.py || python plot_results.py
