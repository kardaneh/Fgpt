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
# Libraries
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

# Every .f90 in SRC_DIR, listed by basename (no directory, no extension)
EXAMPLES := $(notdir $(basename $(wildcard $(SRC_DIR)/*.f90)))

# -----------------------------
# Targets
# -----------------------------
.PHONY: all clean help $(EXAMPLES)
.PHONY: $(addprefix run_,$(EXAMPLES))
.PHONY: $(addprefix run,$(EXAMPLES))

# Default: do nothing but list what is available
all: help

help:
	@echo "Available examples:"
	@for e in $(EXAMPLES); do echo "  make $$e          # build $$e"; \
	                          echo "  make run_$$e      # build and run $$e"; done
	@echo ""
	@echo "  make clean        # remove build artifacts"

# -----------------------------
# Generic compile rule
# -----------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	@mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR)

# -----------------------------
# Generic link rule: target name == source basename
# -----------------------------
$(EXAMPLES): %: $(OBJ_DIR)/%.o
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $^ $(LDFLAGS) -o $@

# -----------------------------
# Run rules
#   make run_example_01   -> build then execute ./example_01
#   make runexample_01    -> same, shorter alias
# -----------------------------
$(addprefix run_,$(EXAMPLES)): run_%: %
	./$<

$(addprefix run,$(EXAMPLES)): run%: %
	./$<

# -----------------------------
# Clean
# -----------------------------
clean:
	rm -rf $(OBJ_DIR) $(MOD_DIR) $(EXAMPLES) *.txt *.dat *.nc *.bin
	@for f in $(SRC_DIR)/*.f90; do \
		base=$$(basename $$f .f90); \
		org="$(SRC_DIR)/$${base}_org.fgpt"; \
		if [ -f "$$org" ]; then \
			echo "Restoring $$f from $$org"; \
			mv "$$org" "$$f"; \
		fi; \
	done
	@echo "Cleaned build artifacts."
