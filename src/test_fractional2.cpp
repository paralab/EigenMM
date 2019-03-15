#include "eigen_mm.h"

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems);

int main(int argc, char *argv[])
{
    Mat K, M, V;
    Vec lambda;
    SolverOptions options;
    eigen_mm solver;

    options.nevt = 100;
    options.splitmaxiters = 10;
    options.nodesperevaluator = 1;
    options.subproblemsperevaluator = 16;
    options.taskspernode = 28;
    options.splittol = 1e-3;
    options.p = 30;
    options.nv = 10;
    options.raditers = 10;
    options.radtol = 1e-3;
    options.L = 0.01;
    options.R = 10000.0;

    SlepcInitialize(NULL,NULL,NULL,NULL);
    PetscPrintf(PETSC_COMM_WORLD, "Slepc has been initialized\n");

    PetscPrintf(PETSC_COMM_WORLD, "Loading global input matrix\n");
    loadMatsFromFile(&K, &M, 2, "square", 1, 49);

    PetscPrintf(PETSC_COMM_WORLD, "Initializing solver\n");
    solver.init(K, M, options);
    PetscPrintf(PETSC_COMM_WORLD, "Running solver\n");
    solver.solve(&V, &lambda);

    PetscPrintf(PETSC_COMM_WORLD, "Finalizing SLEPC\n");

    MatDestroy(&K);
    MatDestroy(&M);
    //MatDestroy(&V);
    //VecDestroy(&lambda);
    SlepcFinalize();

    return 0;
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