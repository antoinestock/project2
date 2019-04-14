/**
 * C program to solve the two-dimensional Poisson equation on
 * a unit square using one-dimensional eigenvalue decompositions
 * and fast sine transforms.
 *
 * Einar M. Rønquist
 * NTNU, October 2000
 * Revised, October 2001
 * Revised by Eivind Fonn, February 2015
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h> 
#include <omp.h> 

#define PI 3.14159265358979323846
#define true 1
#define false 0

typedef double real;
typedef int bool;

// Function prototypes
real *mk_1D_array(size_t n, bool zero);
int  *mk_1D_array_int(size_t n, bool zero);
real **mk_2D_array(size_t n1, size_t n2, bool zero);
real rhs(real x, real y);
real ctest_fun(real x, real y);

void init_MPI(int argc, char** argv, int* rank, int* size, double* start);
void init_transpose(int argc, char** argv, size_t m, int rank, int size, int* displ, int* rows);
void transpose(real **bt, real **b, size_t m, int rank, int size, int *displ, int *rows, int n_thread);
void close_MPI(int rank, double* start,int size, int n_thread, int n);
void get_ctest(real **b, real *grid, int m, int rank, int size, int *displ, int *rows, int n_thread);

// Functions implemented in FORTRAN in fst.f and called from C.
// The trailing underscore comes from a convention for symbol names, called name
// mangling: if can differ with compilers.
void fst_(real *v, int *n, real *w, int *nn);
void fstinv_(real *v, int *n, real *w, int *nn);

int main(int argc, char **argv)
{
	    if (argc < 3) { 
        printf("Usage:mpirun -np p file n, t\n"); 
        printf("  poisson n\n\n");
        printf("Arguments:\n");
        printf("  p: the number of processes\n"); 
        printf("  n: the problem size (must be a power of 2)\n");
        printf("  t: the number of threads\n");  
        return 1;
    }

    /*
     *  The equation is solved on a 2D structured grid and homogeneous Dirichlet
     *  conditions are applied on the boundary:
     *  - the number of grid points in each direction is n+1,
     *  - the number of degrees of freedom in each direction is m = n-1,
     *  - the mesh size is constant h = 1/n.
     */
    int n = atoi(argv[1]);
    if ((n & (n-1)) != 0) {
      printf("n must be a power-of-two\n");
      return 2;
    }
    
	int rank, size; 
	int n_thread = atoi(argv[2]); 

    int m = n - 1;
    real h = 1.0 / n;
    double start;
    init_MPI(argc, argv, &rank, &size, &start); 

    int *rows= mk_1D_array_int(size, false);
    int *displ= mk_1D_array_int(size, false);

    init_transpose(argc, argv, m, rank, size, displ, rows);
    /*
     * Grid points are generated with constant mesh size on both x- and y-axis.
     */
 
    real *grid = mk_1D_array(n+1, false);
#pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < n+1; i++) {
        grid[i] = i * h;
    }
    /*
     * The diagonal of the eigenvalue matrix of T is set with the eigenvalues
     * defined Chapter 9. page 93 of the Lecture Notes.
     * Note that the indexing starts from zero here, thus i+1.
     */
    real *diag = mk_1D_array(m, false);
#pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = 0; i < m; i++) {
        diag[i] = 2.0 * (1.0 - cos((i+1) * PI / n));
    }

    /*
     * Allocate the matrices b and bt which will be used for storing value of
     * G, \tilde G^T, \tilde U^T, U as described in Chapter 9. page 101.
     */
    real **b = mk_2D_array(m, m, false);
    real **bt = mk_2D_array(m, m, false);

    /*
     * This vector will holds coefficients of the Discrete Sine Transform (DST)
     * but also of the Fast Fourier Transform used in the FORTRAN code.
     * The storage size is set to nn = 4 * n, look at Chapter 9. pages 98-100:
     * - Fourier coefficients are complex so storage is used for the real part
     *   and the imaginary part.
     * - Fourier coefficients are defined for j = [[ - (n-1), + (n-1) ]] while 
     *   DST coefficients are defined for j [[ 0, n-1 ]].
     * As explained in the Lecture notes coefficients for positive j are stored
     * first.
     * The array is allocated once and passed as arguments to avoid doings 
     * reallocations at each function call.
     */
    int nn = 4 * n;
    real **z = mk_2D_array(n_thread, nn, false);

    /*
     * Initialize the right hand side data for a given rhs function.
     * 
     */
#pragma omp parallel for num_threads(n_thread) schedule(static) collapse(2)
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            b[i][j] = h * h * rhs(grid[i+1], grid[j+1]);
        }
    }

    /*
     * Compute \tilde G^T = S^-1 * (S * G)^T (Chapter 9. page 101 step 1)
     * Instead of using two matrix-matrix products the Discrete Sine Transform
     * (DST) is used.
     * The DST code is implemented in FORTRAN in fst.f and can be called from C.
     * The array zz is used as storage for DST coefficients and internally for 
     * FFT coefficients in fst_ and fstinv_.
     * In functions fst_ and fst_inv_ coefficients are written back to the input 
     * array (first argument) so that the initial values are overwritten.
     */

#pragma omp parallel for num_threads(n_thread) schedule(static)
     for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
         fst_(b[i], &n, z[omp_get_thread_num()], &nn);
     }

    transpose(bt, b, m, rank, size, displ, rows, n_thread);


#pragma omp parallel for num_threads(n_thread) schedule(static)   
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        fstinv_(bt[i], &n, z[omp_get_thread_num()], &nn);
    }
    /*
     * Solve Lambda * \tilde U = \tilde G (Chapter 9. page 101 step 2)
     */
#pragma omp parallel for num_threads(n_thread) schedule(static) collapse(2)
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = bt[i][j] / (diag[i] + diag[j]);
        }
    }

    /*
     * Compute U = S^-1 * (S * Utilde^T) (Chapter 9. page 101 step 3)
     */
#pragma omp parallel for num_threads(n_thread) schedule(static)
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        fst_(bt[i], &n, z[omp_get_thread_num()], &nn);
    }
    
    transpose(b, bt, m, rank, size, displ, rows, n_thread);
    
#pragma omp parallel for num_threads(n_thread) schedule(static)    
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        fstinv_(b[i], &n, z[omp_get_thread_num()], &nn);
    }
 
#ifdef CT  //convergence test
 	get_ctest(b, grid, m, rank, size, displ, rows, n_thread);
#else
#endif
	close_MPI(rank, &start,size,n_thread,n);
    free(displ);
    free(rows);  
	return 1;
}

/*
 * This function is used for initializing the right-hand side of the equation.
 * Other functions can be defined to swtich between problem definitions.
 */

real rhs(real x, real y) {
#ifdef CT
	return 5*PI*PI*sin(PI*x)*sin(2.0*PI*y);
#else // define source function here
    return 1;
#endif
}
/*
 * This function is used for convergence test.
 */
real ctest_fun(real x, real y) {
	return sin(PI*x)*sin(2.0*PI*y);
}
// calculate error
void get_ctest(real **b, real *grid, int m, int rank, int size, int *displ, int *rows, int n_thread)
{
	double error=0.0;
#ifdef HAVE_MPI
	double global_error=0.0;
#pragma omp parallel for schedule(static) reduction(max:error)
    for (size_t i = displ[rank]; i < displ[rank]+rows[rank]; i++) {
        for (size_t j = 0; j < m; j++) {
            error = error > fabs(ctest_fun(grid[i+1],grid[j+1])-b[i][j]) ? error : fabs(ctest_fun(grid[i+1],grid[j+1])-b[i][j]);
        }
    }
    MPI_Allreduce(&error,&global_error,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    
    if(rank==0){
    printf("global error = %e\n", global_error);
	}
#else
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            error = error > fabs(ctest_fun(grid[i+1],grid[j+1])-b[i][j]) ? error : fabs(ctest_fun(grid[i+1],grid[j+1])-b[i][j]);
        }
    }
    printf("global error = %e\n", error);
#endif
}
/*
 * The allocation of a vectore of size n is done with just allocating an array.
 * The only thing to notice here is the use of calloc to zero the array.
 */

real *mk_1D_array(size_t n, bool zero)
{
    if (zero) {
        return (real *)calloc(n, sizeof(real));
    }
    return (real *)malloc(n * sizeof(real));
}

int *mk_1D_array_int(size_t n, bool zero)
{
    if (zero) {
        return (int *)calloc(n, sizeof(int));
    }
    return (int *)malloc(n * sizeof(int));
}

/*
 * The allocation of the two-dimensional array used for storing matrices is done
 * in the following way for a matrix in R^(n1*n2):
 * 1. an array of pointers is allocated, one pointer for each row,
 * 2. a 'flat' array of size n1*n2 is allocated to ensure that the memory space
 *   is contigusous,
 * 3. pointers are set for each row to the address of first element.
 */

real **mk_2D_array(size_t n1, size_t n2, bool zero)
{
    // 1
    real **ret = (real **)malloc(n1 * sizeof(real *));

    // 2
    if (zero) {
        ret[0] = (real *)calloc(n1 * n2, sizeof(real));
    }
    else {
        ret[0] = (real *)malloc(n1 * n2 * sizeof(real));
    }
    
    // 3
    for (size_t i = 1; i < n1; i++) {
        ret[i] = ret[i-1] + n2;
    }
    return ret;
}

void init_MPI(int argc, char** argv, int* rank, int* size, double *start)
{
#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    *start= MPI_Wtime();
#else
    *rank = 0;
    *size = 1;
#endif
}

void init_transpose(int argc, char** argv, size_t m, int rank, int size, int* displ, int* rows)
{
#ifdef HAVE_MPI

  //number of vector and displacement for each rank
    displ[0]=0;
	for (int i=0;i<size;++i) {
    rows[i] = m/size;
    if (m % size && i >= (size - m % size))
      rows[i]++;
    if (i<size -1)
      displ[i+1]=displ[i]+rows[i];
}
#else
    *displ= 0;
    *rows= m;
#endif
}


/*
 * Write the transpose of b a matrix of R^(m*m) in bt.
 * In parallel the function MPI_Alltoallv is used to map directly the entries
 * stored in the array to the block structure, using displacement arrays.
 */


void transpose(real **bt, real **b, size_t m, int rank, int size, int *displ, int *rows, int n_thread)
{
#ifdef HAVE_MPI
	//setup parameters for MPI_ALLTOALLV
	int *scount= mk_1D_array_int(size, false);
	int *rcount= mk_1D_array_int(size, false);
	int *sdispl = mk_1D_array_int(size, false);
	int *rdispl = mk_1D_array_int(size, false);
			
	MPI_Datatype Vector_col;
	MPI_Datatype Vector_col0;
			
	MPI_Type_vector(m,1,m,MPI_DOUBLE,&Vector_col);
	MPI_Type_create_resized(Vector_col,0,sizeof(double),&Vector_col0);
	MPI_Type_commit(&Vector_col0); 	
 	MPI_Type_free(&Vector_col);
 	
#pragma omp parallel for num_threads(n_thread) schedule(static)   	
	for (int i=0; i < size; i++){ //find index
	scount[i]= m*rows[rank];
	sdispl[i]= m*displ[rank]; 
	rcount[i]= rows[i];  
	rdispl[i]= displ[i];
	}
	
 	MPI_Alltoallv(b[0],scount,sdispl,MPI_DOUBLE,
                  bt[0],rcount,rdispl,Vector_col0,MPI_COMM_WORLD);         

	free(scount);
	free(sdispl);
	free(rcount);
	free(rdispl);
	
#else
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            bt[i][j] = b[j][i];
        }
    }
#endif
}


void close_MPI(int rank, double* start, int size, int n_thread, int n)
{
#ifdef HAVE_MPI
	double time = MPI_Wtime()-*start;
	if (rank==0){
	printf("Time: %f\n", time);
	}
	  FILE *fp;
  fp=fopen("poisson_result.txt","a+");

  if(rank==0)  {
		  fprintf(fp,"procs=%d, thread=%d, n=%d, time=%e\n",size,n_thread,n,time);
		  }
    fclose(fp);
  
	MPI_Finalize();
#endif
}

