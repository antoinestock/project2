#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:15:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=2

#SBATCH --job-name="poisson2"
#SBATCH --output=poisson-p24t2n.out
#SBATCH --mail-user=weiwan@ntnu.no
#SBATCH --mail-type=ALL



WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}

module purge
module load GCC
module load OpenMPI

for i in `seq 1 14`
	do
    mpirun -np 24 ./poisson $[2**$i] 2 
done

