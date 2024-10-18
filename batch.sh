#!/bin/bash
#SBATCH -A EUHPC_D05_042
#SBATCH -p boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --time 00:10:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=423000
#SBATCH --job-name=jobMPI
#SBATCH --err=jobMPI.err
#SBATCH --out=jobMPI.out
source arch.env
mpirun -np 1 ./hydrol/hydrol_soil/hydrol_soil

