#ifndef eigen_mm_H
#define eigen_mm_H

#include <petsc.h>
#include <slepceps.h>

#include <vector>
#include <string>
#include <iostream>
#include "mpi.h"
//#include "zfp.h"

#define __DEBUG1__

#define PI 3.141592653589793238462

struct SolverOptions
{
    PetscInt nevt;
    PetscInt splitmaxiters;
    PetscInt nodesperevaluator;
    PetscInt subproblemsperevaluator;
    PetscInt taskspernode;
    PetscInt p;
    PetscInt nv;
    PetscInt raditers;
    PetscInt nevaluators;
    PetscInt totalsubproblems;
    PetscReal splittol;
    PetscReal radtol;
    PetscReal L;
    PetscReal R;
    bool terse;
    bool details;
    bool debug;
    bool saveoutput;
    const char* output_filepath;
};
struct NodeInfo
{
    // world
    PetscInt worldrank, worldsize;

    // evaluator
    PetscInt rank, size, id;
    MPI_Comm comm;

    // row
    PetscInt rowrank, rowsize, rowid;
    MPI_Comm rowcomm;

    // additional properties
    PetscInt processes_per_node, taskspernode, nevaluators;
    PetscViewer viewer;

    // Results
    PetscInt neval;
};

class eigen_mm{
    // Generalized eigenvalue problem:
    // A v = \lambda B v

private:

    SolverOptions opts;
    NodeInfo node;
    MPI_Comm comm = MPI_COMM_WORLD;
    Mat K, M, V;
    Vec lambda;
    std::vector<PetscReal> intervals;
    std::vector<PetscReal> lambda_data;
    std::vector<PetscReal> lv_data;
    unsigned int eig_num = 0; // number of computed eigenvaleus.

public:

    bool store_eigenpairs = true; // set this to true to store eigenpairs in the wrapper class.
    std::vector<double> eig_val_real; // real part of eigenvalues. the same on all the processors
    std::vector<double> eig_val_imag; // imaginary part of eigenvalues. the same on all the processors
    std::vector< std::vector<double> > eig_vec_real; // real part of eigenvectors. local part of the vector on each processor
    std::vector< std::vector<double> > eig_vec_imag; // imaginary part of eigenvectors. local part of the vector on each processor

    eigen_mm();
    ~eigen_mm();

    int init(Mat &K_in, Mat &M_in, SolverOptions opt);
    Mat& getK();
    Mat& getM();
    int solve(Mat *V_out, Vec *lambda_out);
    void print_eig_val_real();
    void print_eig_val_imag();
    void print_eig_val();
    void print_eig_vec_real(int ran);
    void print_eig_vec_imag(int ran);
    void print_eig_vec(int ran);

    int viewK();
    int viewM();

    int get_eig_num();
    double* get_eig_val_real();
    double* get_eig_val_imag();
    double* get_eig_vec_real(int i);
    double* get_eig_vec_imag(int i);

    void findUpperBound();
    void formSubproblems();
    PetscInt solveSubproblems();
    void formEigenbasis(PetscInt neval);

    void scatterInputMats(Mat &K_in, Mat &M_in);
    PetscInt solveSubProblem(PetscReal a, PetscReal b, int job);
    void splitSubProblem(PetscReal a, PetscReal b, PetscReal *c, 
        PetscInt *out_ec_left, PetscInt *out_ec_right);
    PetscInt computeDev(PetscReal a, PetscReal U, PetscBool rl);
    PetscReal computeRadius(Mat &A);    
    void countInterval(PetscReal a, PetscReal b, PetscInt *count);
};

template<class T>
int print_vector(const std::vector<T> &v, const int ran, const std::string &name, MPI_Comm comm){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), ran, v.size());
            for (auto i:v) {
                std::cout << iter << "\t" << i << std::endl;
                iter++;
            }
            printf("\n");
        }
    } else{
        for(unsigned proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), proc, v.size());
                for (auto i:v) {
                    std::cout << iter << "\t" << i << std::endl;
                    iter++;
                }
                printf("\n");
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}

//double print_time(double t_start, double t_end, const std::string function_name, MPI_Comm comm);

#endif //eigen_mm_H
