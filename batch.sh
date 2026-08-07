#!/usr/bin/env bash
#
#
#SBATCH -A EUHPC_D05_042
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1                # modern Slurm; if your cluster uses gres, change to --gres=gpu:1
#SBATCH --mem=423G              # explicit units (423 GB)
#SBATCH --job-name=hydrol_job
#SBATCH --output=jobMPI.%j.out
#SBATCH --error=jobMPI.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.tld   # <-- update this

set -euo pipefail
trap 'echo "ERROR on line $LINENO"; exit 1' ERR

# Move to submission directory so relative paths work
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Load environment / modules — replace with your cluster's module commands
if command -v module &>/dev/null; then
  module purge
fi

# Source local environment if present
if [[ -f arch.env ]]; then
  # arch.env should set PATH, LD_LIBRARY_PATH, conda env, etc.
  source arch-nvhpc.env
fi

# Path to executable (use absolute or ensure PATH is correct)
EXE=./hydrol/hydrol_soil/hydrol_soil

if [[ ! -x "$EXE" ]]; then
  echo "Executable $EXE not found or not executable."
  echo "Listing directory:"
  ls -l "$(dirname "$EXE")" || true
  exit 1
fi

# Prefer srun (integrates with Slurm). If you must use mpirun, ensure it is
# the Slurm-wrapped mpirun or the MPI implementation used by your system.
# Example with srun:
srun --mpi=pmix "$EXE"

# Alternative (uncomment if required by your system):
# mpirun -np 1 "$EXE"
