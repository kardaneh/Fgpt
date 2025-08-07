WORK=/data/ssivanes

FC_NVIDIA = mpif90

NETCDF_INCDIR = -I$(NETCDF_C_ROOT)/include -I$(NETCDF_FORTRAN_ROOT)/include
NETCDF_LIBDIR = -L$(NETCDF_C_ROOT)/lib -lnetcdf -L$(NETCDF_FORTRAN_ROOT)/lib -lnetcdff

IOIPSL_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/IOIPSL/inc
IOIPSL_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/IOIPSL/lib -lioipsl

XIOS_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/XIOS/inc
XIOS_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/XIOS/lib -lxios -lstdc++

ORCHIDEE_INCDIR = -I$(WORK)/modipsl_truck_opt/modeles/ORCHIDEE/inc
ORCHIDEE_LIBDIR = -L$(WORK)/modipsl_truck_opt/modeles/ORCHIDEE/lib -lorchidee

FFLAGS = $(IOIPSL_INCDIR) $(XIOS_INCDIR) $(ORCHIDEE_INCDIR) $(NETCDF_INCDIR)
LDFLAGS = $(ORCHIDEE_LIBDIR) $(IOIPSL_LIBDIR) $(XIOS_LIBDIR) $(NETCDF_LIBDIR)
FFLAGS_COMMON = -Wall -O0 -Kieee -Ktrap=fp #-Mbounds
FFLAGS_CPU = $(FFLAGS_COMMON) -r8 -i4
FFLAGS_GPU = $(FFLAGS_COMMON) -r8 -i4 -Minfo=acc -ta=tesla:ccall,cc80 -acc=noautopar
	
ifeq ($(strip $(MODE)),GPU)
	FFLAGS_NVIDIA = $(FFLAGS_GPU)
else
	FFLAGS_NVIDIA = $(FFLAGS_CPU)
endif
#FFLAGS_NVIDIA = $(FFLAGS_COMMON) -r8 -i4 #-Minfo=acc -ta=tesla:ccall,cc80 -acc=noautopar #-gpu=managed:ccall -acc=noautopar #-ta=tesla -acc=noautopar

OBJ_DIR = obj
MOD_DIR = mod

SRC_DIR = $(SUBDIR_PATH)
MODULE_SRC = $(wildcard $(SRC_DIR)/*.f90)
MODULE_OBJ = $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%,$(MODULE_SRC:.f90=.o))

EXECUTABLE = $(notdir $(CURDIR))
COMPILER_REPORT = $(SRC_DIR)/$(EXECUTABLE).txt

.PHONY: all clean
all: clean $(EXECUTABLE)

$(EXECUTABLE): $(MODULE_OBJ)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $^ $(LDFLAGS) -o $@ >> $(COMPILER_REPORT) 2>&1

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.f90
	mkdir -p $(OBJ_DIR) $(MOD_DIR)
	$(FC_NVIDIA) $(FFLAGS_NVIDIA) $(FFLAGS) -c $< -o $@ -module $(MOD_DIR) >> $(COMPILER_REPORT) 2>&1

clean:
	rm -rf $(OBJ_DIR) $(EXECUTABLE) $(MOD_DIR) $(COMPILER_REPORT)

