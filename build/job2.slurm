#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2

#SBATCH --job-name="poisson"
#SBATCH --output=poisson-p2t18n.out
#SBATCH --mail-user=weiwan@ntnu.no
#SBATCH --mail-type=ALL



WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load GCC
module load OpenMPI

mpirun -np 2 ./poisson 16384 2 

