#ifndef eigen_mm_H
#define eigen_mm_H

#include <petsc.h>
#include <slepceps.h>

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <iterator>
#include <iostream>
#include "mpi.h"

#define __DEBUG1__

#define PI 3.141592653589793238462

class SolverOptions
{
private:
    PetscInt _nevt = 100;
    PetscInt _splitmaxiters = 10;
    PetscInt _nodesperevaluator = 1;
    PetscInt _subproblemsperevaluator = 1;
    PetscInt _totalsubproblems = 1;
    PetscInt _nevaluators = 1;
    PetscInt _p = 30;
    PetscInt _nv = 10;
    PetscInt _raditers = 10;
    PetscReal _splittol = 0.9;
    PetscReal _radtol = 1e-3;
    PetscReal _L = 0.01;
    PetscReal _R = 10000.0;
    bool _terse = false;
    bool _details = false;
    bool _debug = false;
    bool _save_correctness = false;
    bool _save_operators = false;
    bool _save_eigenvalues = false;
    bool _save_eigenbasis = false;
    const char* _correctness_filename = "";
    const char* _operators_filename = "";
    const char* _eigenvalues_filename = "";
    const char* _eigenbasis_filename = "";
    PetscInt _eps_solver_type = 0;
    PetscInt _ksp_solver_type = 0;

public:
    SolverOptions() {}

    // setters
    void set_nevt(PetscInt v) { _nevt = v; }
    void set_splitmaxiters(PetscInt v) { _splitmaxiters = v; }
    void set_nodesperevaluator(PetscInt v) { _nodesperevaluator = v; }
    void set_subproblemsperevaluator(PetscInt v) { _subproblemsperevaluator = v; }
    void set_nevaluators(PetscInt v) { _nevaluators = v; }
    void set_p(PetscInt v) { _p = v; }
    void set_nv(PetscInt v) { _nv = v; }
    void set_raditers(PetscInt v) { _raditers = v; }
    void set_splittol(PetscReal v) { _splittol = v; }
    void set_radtol(PetscReal v) { _radtol = v; }
    void set_L(PetscReal v) { _L = v; }
    void set_R(PetscReal v) { _R = v; }
    void set_terse(bool v) { _terse = v; }
    void set_details(bool v) { _details = v; }
    void set_debug(bool v) { _debug = v; }
    void set_save_correctness(bool v, const char* filename)
    {
        _save_correctness = v;
        _correctness_filename = filename;
    }
    void set_save_operators(bool v, const char* filename)
    {
        _save_operators = v;
        _operators_filename = filename;
    }
    void set_save_eigenvalues(bool v, const char* filename) 
    {
        _save_eigenvalues = v;
        _eigenvalues_filename = filename; 
    }
    void set_save_eigenbasis(bool v, const char* filename)
    {
        _save_eigenbasis = v;
        _eigenbasis_filename = filename;
    }
    void set_totalsubproblems(PetscInt v) { _totalsubproblems = v; }
    void set_eps_solver_type(PetscInt v) { _eps_solver_type = v; }
    void set_ksp_solver_type(PetscInt v) { _ksp_solver_type = v; }
    
    // getters
    PetscInt nevt() { return _nevt; }
    PetscInt splitmaxiters() { return _splitmaxiters; }
    PetscInt nodesperevaluator() { return _nodesperevaluator; }
    PetscInt subproblemsperevaluator() { return _subproblemsperevaluator; }
    PetscInt nevaluators() { return _nevaluators; }
    PetscInt p() { return _p; }
    PetscInt nv() { return _nv; }
    PetscInt raditers() { return _raditers; }
    PetscReal splittol() { return _splittol; }
    PetscReal radtol() { return _radtol; }
    PetscReal L() { return _L; }
    PetscReal R() { return _R; }
    bool terse() { return _terse; }
    bool details() { return _details; }
    bool debug() { return _debug; }
    bool save_correctness() { return _save_correctness; }
    bool save_operators() { return _save_operators; }
    bool save_eigenvalues() { return _save_eigenvalues; }
    bool save_eigenbasis() { return _save_eigenbasis; }
    const char* correctness_filename() { return _correctness_filename; }
    const char* operators_filename() { return _operators_filename; }
    const char* eigenvalues_filename() { return _eigenvalues_filename; }
    const char* eigenbasis_filename() { return _eigenbasis_filename; }
    PetscInt totalsubproblems() { return _totalsubproblems; }
    PetscInt eps_solver_type() { return _eps_solver_type; }
    PetscInt ksp_solver_type() { return _ksp_solver_type; }
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
    PetscInt nevaluators;
    PetscViewer viewer;

    // Results
    PetscInt neval;
    PetscInt neval0;
};

class eigen_mm{
    // Generalized eigenvalue problem:
    // A v = \lambda B v

private:

    EPS eps;
    SolverOptions opts;
    NodeInfo node;
    MPI_Comm comm = MPI_COMM_WORLD;
    Mat K, M, V;
    Mat K_global, M_global;
    Vec lambda;
    std::vector<PetscReal> intervals;
    std::vector<PetscReal> residuals;
    unsigned int eig_num = 0; // number of computed eigenvaleus.

    PetscReal find_amax(PetscInt k);
    PetscReal find_b(PetscInt k, PetscReal a);

public:

    bool store_eigenpairs = true; // set this to true to store eigenpairs in the wrapper class.
    std::vector<double> eig_val_real; // real part of eigenvalues. the same on all the processors
    std::vector<double> eig_val_imag; // imaginary part of eigenvalues. the same on all the processors
    std::vector< std::vector<double> > eig_vec_real; // real part of eigenvectors. local part of the vector on each processor
    std::vector< std::vector<double> > eig_vec_imag; // imaginary part of eigenvectors. local part of the vector on each processor

    eigen_mm();
    ~eigen_mm();

    int init(Mat &K_in, Mat &M_in, SolverOptions *opt);
    Mat& getK();
    Mat& getM();
    int solve(Mat *V_out, Vec *lambda_out);
    int solvetime_exp();
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

    void checkCorrectness();
    void checkOrthogonality();

    void findUpperBound();
    void formSubproblems();
    PetscInt solveSubproblems();
    void formEigenbasis(PetscInt neval);

    void scatterInputMats(Mat &K_in, Mat &M_in);
    PetscInt solveSubProblem(PetscReal *intervals, int job);
    void splitSubProblem(PetscReal a, PetscReal b, PetscReal *c, 
        PetscInt *out_ec_left, PetscInt *out_ec_right);
    PetscInt computeDev_approximate(PetscReal a, PetscReal U, PetscBool rl);
    PetscInt computeDev_exact(PetscReal a, PetscBool rl);
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
