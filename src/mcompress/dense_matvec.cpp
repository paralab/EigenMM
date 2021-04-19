#include <mpi.h>
#include <mkl.h>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>
#include <petsc.h>

using std::cout;
using std::endl;

typedef std::chrono::high_resolution_clock Clock;

const double NS2MS = 0.000001;

void loadMatsFromFile(Mat *V, const char* filename)
{
    PetscViewer viewer;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, V);
    MatSetType(*V, MATMPIDENSE);
    MatLoad(*V, viewer);
    MatAssemblyBegin(*V, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*V, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);
}

void randomUnitVector(Vec &x, PetscRandom &r, int rank)
{
    VecSetRandom(x, r);

    double nrm = 0.0;
    VecNorm(x, NORM_2, &nrm);

    VecScale(x, 1.0 / nrm);
}

void clearVector(Vec &x)
{
    VecSet(x, 0.0);
}

void computeStatistics( std::vector<double> vals, double *mean, double *var)
{
    double lmean = 0.0;
    double lvar = 0.0;
    for (int i = 0; i < vals.size(); i++)
    {
        double lmean_update = lmean + (vals[i] - lmean)/(i+1);
        lvar = lvar + (vals[i] - lmean)*(vals[i] - lmean_update);
        lmean = lmean_update;
    }

    mean[0] = lmean;
    var[0] = lvar;
}

void timingExperiment( Mat &A, int nk )
{
    Vec x, b;

    std::vector<double> elapsed(nk);

    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int m, n;
    MatGetSize(A, &m, &n);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &x);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, m, &b);

    PetscRandom r;
    PetscRandomCreate(PETSC_COMM_WORLD, &r);
    PetscRandomSetType(r, PETSCRAND);

    for (int i = 0; i < nk; i++)
    {
        // generate random unit vector x
        randomUnitVector(x, r, rank);

        // clear b
        clearVector(b);

        // start timer
        MPI_Barrier(PETSC_COMM_WORLD);
        auto start_time = Clock::now();

        // multiply A*x = b
        MatMult(A, x, b);

        // stop timer
        MPI_Barrier(PETSC_COMM_WORLD);
        auto stop_time = Clock::now();

        // store multiplication time
        elapsed[i] = NS2MS * 
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
    }

    double mean = 0.0;
    double var = 0.0;
    computeStatistics(elapsed, &mean, &var);
    if (rank == 0) 
        printf("Matrix A (%d by %d) multiplication time took roughly %lf +/- %lf milliseconds\n", m, n, mean, sqrt(var/nk));
}

int main ( int argc, char* argv[] )
{    
    Mat V;

    int nk;
    const char* eigenbasis_filename;
    if (argc < 2)
    {
        printf("Provide eigenbasis filename.\n");
        return 0;
    }
    else if (argc < 3)
    {
        eigenbasis_filename = argv[1];
        nk = 100;
    }
    else
    {
        eigenbasis_filename = argv[1];
        nk = atoi(argv[2]);
    }

    // Initialize SLEPc
    PetscInitialize(NULL, NULL, NULL, NULL);

    // Load eigenbasis
    loadMatsFromFile(&V, eigenbasis_filename);

    // Run matvec timing experiment
    timingExperiment(V, nk);

    PetscFinalize();
}