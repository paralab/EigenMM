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
#include "tinyxml.h"

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
    PetscInt _nevals = -1;
    PetscInt _nk = 5;
    PetscInt _nb = 3;
    PetscInt _p = 30;
    PetscInt _nv = 10;
    PetscInt _raditers = 10;
    PetscReal _splittol = 0.9;
    PetscReal _radtol = 1e-3;
    PetscReal _L = 0.01;
    PetscReal _R = -1.0;
    bool _terse = false;
    bool _details = false;
    bool _debug = false;
    bool _save_operators = false;
    bool _save_correctness = false;
    bool _save_eigenvalues = false;
    bool _save_eigenbasis = false;
    char _operators_filename[2048];
    char _correctness_filename[2048];
    char _eigenvalues_filename[2048];
    char _eigenbasis_filename[2048];
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
    void set_nevals(PetscInt v) { _nevals = v; }
    void set_nk(PetscInt v) { _nk = v; }
    void set_nb(PetscInt v) { _nb = v; }
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
    void set_save_operators(bool v, const char* filename)
    {
        _save_operators = v;
        sprintf(_operators_filename, "%s", filename);
    }
    void set_save_correctness(bool v, const char* filename)
    {
        _save_correctness = v;
        sprintf(_correctness_filename, "%s", filename);
    }
    void set_save_eigenvalues(bool v, const char* filename) 
    {
        _save_eigenvalues = v;
        sprintf(_eigenvalues_filename, "%s", filename);
    }
    void set_save_eigenbasis(bool v, const char* filename)
    {
        _save_eigenbasis = v;
         sprintf(_eigenbasis_filename, "%s", filename);
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
    PetscInt nevals() { return _nevals; }
    PetscInt nk() { return _nk; }
    PetscInt nb() { return _nb; }
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
    bool save_operators() { return _save_operators; }
    bool save_correctness() { return _save_correctness; }
    bool save_eigenvalues() { return _save_eigenvalues; }
    bool save_eigenbasis() { return _save_eigenbasis; }
    char* operators_filename() { return _operators_filename; }
    char* correctness_filename() { return _correctness_filename; }
    char* eigenvalues_filename() { return _eigenvalues_filename; }
    char* eigenbasis_filename() { return _eigenbasis_filename; }
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

    // additional properties
    PetscInt nevaluators;
    
    const char* procname;
    PetscInt procid;

    // Results
    PetscInt neval;
    PetscInt neval0;
};

class eigen_mm
{
private:

    EPS eps;
    SolverOptions opts;
    NodeInfo node;
    Mat K, M, V;
    Mat K_global, M_global;
    Vec lambda;
    std::vector<PetscReal> intervals;
    std::vector<PetscReal> residuals;

public:

    eigen_mm();
    ~eigen_mm();

    int init(Mat &K_in, Mat &M_in, SolverOptions &opt);
    int solve(Mat *V_out, Vec *lambda_out);

    void findUpperBound();
    void rescaleInterval();
    void formSubproblems();
    PetscInt solveSubproblems();
    void formEigenbasis(PetscInt neval);

    PetscInt solveSubProblem(PetscReal *intervals, int job);
    void splitSubProblem(PetscReal a, PetscReal b, PetscReal *c, 
        PetscInt *out_ec_left, PetscInt *out_ec_right);
    void global_refine(PetscInt n,
                       std::vector<PetscReal> &x, 
                       std::vector<PetscInt>   C, 
                       PetscInt Nhat);
    void balance_intervals(PetscReal   a, PetscReal   b, PetscReal   *c, 
                           PetscInt reva, PetscInt revb, PetscInt *revc,
                           PetscInt  *Cl, PetscInt  *Cr);
    PetscInt computeDev_approximate(PetscReal a, PetscReal U, PetscBool rl);
    PetscInt computeDev_exact(PetscReal a, PetscBool rl);
    PetscReal computeRadius(Mat &A);
    void countInterval(PetscReal a, PetscReal b, PetscInt *count);

};

#endif //eigen_mm_H