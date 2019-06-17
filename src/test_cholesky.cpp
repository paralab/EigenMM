#include "eigen_mm.h"
#include <limits>

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems);
PetscReal findupperbound(Mat &K, Mat &M);

int main(int argc, char *argv[])
{
    double dbl_max = std::numeric_limits<double>::max();

    Mat K, M;
    SlepcInitialize(&argc,&argv,NULL,NULL);

    // geometry parameters
    PetscInt dim, nelems, order, nshifts, nsamples;
    char dtype[1024];
    char output_filepath[1024];
    PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL);
    PetscOptionsGetInt(NULL, NULL, "-n", &nelems, NULL);
    PetscOptionsGetInt(NULL, NULL, "-k", &order, NULL);
    PetscOptionsGetString(NULL, NULL, "-D", dtype, 1024, NULL);
    PetscOptionsGetInt(NULL, NULL, "-nshifts", &nshifts, NULL);
    PetscOptionsGetInt(NULL, NULL, "-nsamples", &nsamples, NULL);

    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Load system from file
    PetscPrintf(PETSC_COMM_WORLD, "Loading initial system\n");
    loadMatsFromFile(&K, &M, dim, dtype, order, nelems);

    PetscReal L = 0.0;
    PetscReal R = findupperbound(K, M);
    PetscReal width = (R - L);
    PetscReal step = (0.9 * width) / (nsamples - 1);
    PetscReal shift = 0.05*width;

    std::vector<double> all_avg_elapsed(nsamples);
    double minelapsed = dbl_max;
    double maxelapsed = 0;
    double avgelapsed = 0;
    int count = 0;
    for (int i = 0; i < nshifts; i++)
    {
        for (int j = 0; j < nsamples; j++)
        {
            // A = K - shift*M
            Mat A, L;
            MatConvert(M, MATSAME, MAT_INITIAL_MATRIX, &A);
            MatAYPX(A, -shift, K, DIFFERENT_NONZERO_PATTERN);

            MPI_Barrier(PETSC_COMM_WORLD);
            double exp_start_time = MPI_Wtime();

            MatGetFactor(A, "mumps", MAT_FACTOR_CHOLESKY, &L);
            MatMumpsSetIcntl(L, 13, 1);
            MatCholeskyFactorSymbolic(L, A, 0, 0);
            MatCholeskyFactorNumeric(L, A, 0);

            MPI_Barrier(PETSC_COMM_WORLD);
            double exp_end_time = MPI_Wtime();
            double exp_total_elapsed = exp_end_time - exp_start_time;

            MatDestroy(&L);
            MatDestroy(&A);

            all_avg_elapsed[i] += 1/(j+1) * (exp_total_elapsed - all_avg_elapsed[i]);
            avgelapsed += 1/(count+1) * (exp_total_elapsed - avgelapsed);
            minelapsed = std::min(minelapsed, exp_total_elapsed);
            maxelapsed = std::max(maxelapsed, exp_total_elapsed);
            count++;
        }
        shift += step;
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_elapsed = end_time - start_time;

    PetscPrintf(PETSC_COMM_WORLD, "Total Elapsed: %lf\n", total_elapsed);
    PetscPrintf(PETSC_COMM_WORLD, "Experiment Details:\n");
    PetscPrintf(PETSC_COMM_WORLD, "  Number of shifts:            %d\n",  nshifts);
    PetscPrintf(PETSC_COMM_WORLD, "  Number of samples per shift: %d\n",  nsamples);
    PetscPrintf(PETSC_COMM_WORLD, "  Maximum elapsed:             %lf\n", maxelapsed);
    PetscPrintf(PETSC_COMM_WORLD, "  Minimum elapsed:             %lf\n", minelapsed);
    PetscPrintf(PETSC_COMM_WORLD, "  Average elapsed:             %lf\n", avgelapsed);
    PetscPrintf(PETSC_COMM_WORLD, "  All elapsed:                 [");
    for (int i = 0; i < nshifts-1; i++)
        PetscPrintf(PETSC_COMM_WORLD, "%lf ", all_avg_elapsed[i]);
    PetscPrintf(PETSC_COMM_WORLD, "%lf]\n", all_avg_elapsed[nshifts-1]);

    MatDestroy(&K);
    MatDestroy(&M);
    SlepcFinalize();

    return 0;
}

PetscReal findupperbound(Mat &K, Mat &M)
{
    PetscReal lR;

    Mat F;
    KSP ksp;
    PC pc;
    Vec x, Sx, b;
    PetscRandom r;
    PetscReal e, e0, normx, normsx;
    PetscInt N, iter;

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, M, M);
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCCHOLESKY);
    PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
    PCFactorSetUpMatSolverPackage(pc);
    PCFactorGetMatrix(pc, &F);
    MatMumpsSetIcntl(F, 13, 1);
    MatMumpsSetIcntl(F, 14, 80);

    PetscRandomCreate(PETSC_COMM_WORLD, &r);
    PetscRandomSetType(r, PETSCRAND);
    PetscRandomSetInterval(r, -1, 1);

    MatGetSize(K, &N, NULL);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &x);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &Sx);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &b);

    // initialize v (stored in Sx)
    VecSetRandom(x, r);
    VecCopy(x, Sx);
    VecAbs(Sx);
    VecPointwiseDivide(Sx, x, Sx);
    
    // x = abs( A' * Sx );
    // x = abs( K * (M \ Sx) );
    KSPSolve(ksp, Sx, b);
    MatMult(K, b, x);
    VecAbs(x);
    
    VecNorm(x, NORM_2, &e);
    VecScale(x, 1/e);
    e0 = 0;
    iter = 1;
    while (abs(e - e0) > 1e-3*e && iter <= 10)
    {
        e0 = e;
        // Sx = A * x
        // Sx = M \ (K * x)
        MatMult(K, x, b);
        KSPSolve(ksp, x, Sx);

        // x = A' * Sx
        // x = K * (M \ Sx)
        KSPSolve(ksp, Sx, b);
        MatMult(K, b, x);

        VecNorm(x, NORM_2, &normx);
        VecNorm(Sx, NORM_2, &normsx);
        e = normx / normsx;
        VecScale(x, 1/normx);
        
        iter++;
    }

    lR = e;

    VecDestroy(&x);
    VecDestroy(&Sx);
    VecDestroy(&b);
    KSPDestroy(&ksp);

    return lR;
}

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems)
{
    PetscViewer viewer;
    char filename[512];

    sprintf(filename, "/scratch/kingspeak/serial/u0450449/fractional/matrices/%dD/%s/%d/stiffness_%d", dim, dtype, order, nelems);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, K);
    MatLoad((*K), viewer);
    MatSetOption((*K), MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin((*K), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*K), MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);

    sprintf(filename, "/scratch/kingspeak/serial/u0450449/fractional/matrices/%dD/%s/%d/mass_%d", dim, dtype, order, nelems);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, M);
    MatLoad((*M), viewer);
    MatSetOption((*M), MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin((*M), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*M), MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);
}
