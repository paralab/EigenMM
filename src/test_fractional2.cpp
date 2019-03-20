#include "eigen_mm.h"

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems);

int main(int argc, char *argv[])
{
    Mat V;
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

    solver.init(options);
    solver.loadMatsFromFile(2, "square", 1, 49);
    // solver.loadMatsFromNektar(-- parameters --);
    solver.solve(&V, &lambda);

    return 0;
}

