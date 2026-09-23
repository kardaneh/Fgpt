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

# -----------------------------
# Libraries (NetCDF kept for consistency; not used by this example)
# -----------------------------
NETCDF_INCDIR = -I$(NETCDF_C_ROOT)/include -I$(NETCDF_FORTRAN_ROOT)/include
NETCDF_LIBDIR = -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETCDF_FORTRAN_ROOT)/lib -lnetcdff

# -----------------------------
# Flags
# -----------------------------
FFLAGS  = $(NETCDF_INCDIR)
LDFLAGS = $(NETCDF_LIBDIR)

FFLAGS_COMMON = -Wall -g -O0 -Kieee -Ktrap=fp -Mbounds -traceback
FFLAGS_CPU    = $(FFLAGS_COMMON) -r8 -i4
FFLAGS_NVIDIA = $(FFLAGS_CPU)

# -----------------------------
# Directories and sources
# -----------------------------
OBJ_DIR ?= obj
MOD_DIR ?= mod
SRC_DIR ?= .

# All sources of this project (order matters only on the very first build;
# the explicit prerequisites below enforce the correct order anyway).
SRC = \
  $(SRC_DIR)/mod_parameters.f90 \
  $(SRC_DIR)/mod_grid.f90 \
  $(SRC_DIR)/mod_math.f90 \
  $(SRC_DIR)/mod_io.f90 \
  $(SRC_DIR)/mod_simulation.f90 \
  $(SRC_DIR)/main_program.f90

OBJ = $(patsubst $(SRC_DIR)/%.f90,$(OBJ_DIR)/%.o,$(SRC))

EXECUTABLE = multi_module_demo

# -----------------------------
# Targets
# -----------------------------
.PHONY: all clean help run

all: $(EXECUTABLE)

help:
	@echo "Targets:"
	@echo "  make            # build $(EXECUTABLE)"
	@echo "  make run        # build and execute"
	@echo "  make clean      # remove obj/, mod/, executable, output files"
	@echo "  make restore    # restore any .f90 file that has a matching _org.fgpt"

$(EXECUTABLE): $(OBJ)
	@echo "Linking $(EXECUTABLE) ..."
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $^ $(LDFLAGS) -o $@
	@echo "Build complete: $(EXECUTABLE)"

run: $(EXECUTABLE)
	./$(EXECUTABLE)

# -----------------------------
# Compilation rules (explicit dependency graph)
# -----------------------------
$(OBJ_DIR)/mod_parameters.o: $(SRC_DIR)/mod_parameters.f90
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

$(OBJ_DIR)/mod_grid.o: $(SRC_DIR)/mod_grid.f90 $(OBJ_DIR)/mod_parameters.o
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

$(OBJ_DIR)/mod_math.o: $(SRC_DIR)/mod_math.f90 $(OBJ_DIR)/mod_parameters.o
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

$(OBJ_DIR)/mod_io.o: $(SRC_DIR)/mod_io.f90 $(OBJ_DIR)/mod_parameters.o
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

$(OBJ_DIR)/mod_simulation.o: $(SRC_DIR)/mod_simulation.f90 \
                             $(OBJ_DIR)/mod_parameters.o \
                             $(OBJ_DIR)/mod_grid.o \
                             $(OBJ_DIR)/mod_math.o \
                             $(OBJ_DIR)/mod_io.o
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

$(OBJ_DIR)/main_program.o: $(SRC_DIR)/main_program.f90 \
                           $(OBJ_DIR)/mod_parameters.o \
                           $(OBJ_DIR)/mod_grid.o \
                           $(OBJ_DIR)/mod_math.o \
                           $(OBJ_DIR)/mod_simulation.o
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

# -----------------------------
# Restore sources from .fgpt backups (kept from the reference Makefile)
# -----------------------------
.PHONY: restore
restore:
	@for f in $(SRC_DIR)/*.f90; do \
		base=$$(basename $$f .f90); \
		org="$(SRC_DIR)/$${base}_org.fgpt"; \
		if [ -f "$$org" ]; then \
			echo "Restoring $$f from $$org"; \
			mv "$$org" "$$f"; \
		fi; \
	done

# -----------------------------
# Clean
# -----------------------------
clean:
	rm -rf $(OBJ_DIR) $(MOD_DIR) $(EXECUTABLE)
	rm -f *.txt *.bin
	@echo "Cleaned build artifacts."
