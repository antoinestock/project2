The program is builded with a out-of-tree structure using CMake build system. 
The CMake setup has options for compiling with MPI and OpenMP. 
'Build' folder contents main program. 'Debug' folder is for the convergence test.

One can generate a build system in the folder with the CMakeLists.txt by using:
module load GCC
module load OpenMPI
CC=mpicc FC=mpif90
cmake . -DCMAKE_BUILD_TYPE=Release

Likewise, one can generate a Debug system using: 
cmake . -DCMAKE_BUILD_TYPE=Debug

//////////////////////////////////////////////////////////////////////////////
Makefiles generate in build/Debug folder, and one can then build and run the program using:
make
mpirun -np nprocs ./poisson n nthd

where nprocs is the number of processes
nthd is the number of threads
n is the size of the system; note that n must be a power of two due to the DST.

Likewise, in Debug folder, one can run the program using:

make
Mpirun -np nprocs ./poisson n nthd  -DCT

/////////////////////////////////////////////////////////////////////////////
Slurm folder contents all the job files run on the supercomputer. 
The setup with a different process/thread is defined and run as an individual job.
The slurm file is named indicating its internal setup, for example, jobp12t3n14.slurm meaning
Process: 12
Thread per process: 3
size of the system: 2^14

All the results are written in the poisson_result.txt file.

/////////////////////////////////////////////////////////////////////////////
The results in poisson_result.txt are re-organized in job1.txt file.
poisson_err.m and poisson_speedup.m are used to plot the figures in the report.







