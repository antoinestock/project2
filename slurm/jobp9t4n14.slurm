#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH --job-name="poisson2"
#SBATCH --output=poisson-p9t4n14.out
#SBATCH --mail-user=weiwan@ntnu.no
#SBATCH --mail-type=ALL



WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load GCC
module load OpenMPI
module load CMake

#cmake . -DCMAKE_BUILD_TYPE=Release
#make

mpirun -np 9 ./poisson 16384 4 

