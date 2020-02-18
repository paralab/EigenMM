///////////////////////////////////////////////////////////////////////////////
//
// File eigen_mm.h
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// License for the specific language governing rights and limitations under
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description: EigenMM library definitions
//
///////////////////////////////////////////////////////////////////////////////

#ifndef eigen_mm_H
#define eigen_mm_H

#include <petsc.h>
#include <slepceps.h>

#include <algorithm>
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
    // communicator options
    PetscInt _nodesperevaluator = 1;
    PetscInt _subproblemsperevaluator = 1;
    PetscInt _totalsubproblems = 1;
    PetscInt _nevaluators = 1;
    PetscInt _nevals = -1;
    PetscInt _taskspernode = 28;

    // eigenvalue partitioning options
    PetscInt _nk = 7;
    PetscInt _nb = 4;
    PetscInt _p = 0;
    PetscInt _nv = 10;
    PetscInt _splitmaxiters = 10;
    PetscInt _raditers = 10;
    PetscReal _splittol = 0.9;
    PetscReal _radtol = 1e-3;
    PetscReal _L = 0.01;
    PetscReal _R = -1.0;

    // save and print options
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

public:
    SolverOptions() {}

    // set communicator options
    void set_nodesperevaluator(PetscInt v) { _nodesperevaluator = v; }
    void set_subproblemsperevaluator(PetscInt v) { _subproblemsperevaluator = v; }
    void set_totalsubproblems(PetscInt v) { _totalsubproblems = v; }
    void set_nevaluators(PetscInt v) { _nevaluators = v; }
    void set_nevals(PetscInt v) { _nevals = v; }
    void set_taskspernode(PetscInt v) { _taskspernode = v; }

    // set eigenvalue partitioning options
    void set_nk(PetscInt v) { _nk = v; }
    void set_nb(PetscInt v) { _nb = v; }
    void set_p(PetscInt v) { _p = v; }
    void set_nv(PetscInt v) { _nv = v; }
    void set_splitmaxiters(PetscInt v) { _splitmaxiters = v; }
    void set_raditers(PetscInt v) { _raditers = v; }
    void set_splittol(PetscReal v) { _splittol = v; }
    void set_radtol(PetscReal v) { _radtol = v; }
    void set_L(PetscReal v) { _L = v; }
    void set_R(PetscReal v) { _R = v; }

    // set save and print options
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
    
    // get communicator options
    PetscInt nodesperevaluator() { return _nodesperevaluator; }
    PetscInt subproblemsperevaluator() { return _subproblemsperevaluator; }
    PetscInt totalsubproblems() { return _totalsubproblems; }
    PetscInt nevaluators() { return _nevaluators; }
    PetscInt nevals() { return _nevals; }
    PetscInt taskspernode() { return _taskspernode; }

    // get eigenvalue partitioning options
    PetscInt nk() { return _nk; }
    PetscInt nb() { return _nb; }
    PetscInt p() { return _p; }
    PetscInt nv() { return _nv; }
    PetscInt splitmaxiters() { return _splitmaxiters; }
    PetscInt raditers() { return _raditers; }
    PetscReal splittol() { return _splittol; }
    PetscReal radtol() { return _radtol; }
    PetscReal L() { return _L; }
    PetscReal R() { return _R; }

    // get save and print options
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

    void findUpperBound();
    void rescaleInterval();
    void formSubproblems();
    PetscReal count_subintervals(PetscInt n, PetscReal Nhat, PetscInt Nbar,
        std::vector<PetscReal> x, std::vector<PetscInt> &RA, std::vector<PetscInt> &A);
    void merge_approximations(PetscReal Nhat, PetscInt Nbar,
        std::vector<PetscReal> &x, std::vector<PetscInt> &RA,
        std::vector<PetscReal> y, std::vector<PetscInt> RB,
        std::vector<PetscInt> &A);
    PetscInt solveSubproblems();
    void formEigenbasis(PetscInt neval);

    PetscInt solveSubProblem(PetscReal *intervals, int job);
    void splitSubProblem(PetscReal a, PetscReal b, PetscReal *c, 
        PetscInt *out_ec_left, PetscInt *out_ec_right);
    void global_refine(PetscInt n,
                       std::vector<PetscReal>  x, 
                       std::vector<PetscReal> &y,
                       std::vector<PetscInt>   C, 
                       PetscReal Nhat);
    void balance_intervals(PetscReal   a, PetscReal   b, PetscReal   *c, 
                           PetscInt reva, PetscInt revb, PetscInt *revc,
                           PetscInt  *Cl, PetscInt  *Cr);
    PetscInt computeDev_approximate(PetscReal a, PetscReal U, PetscBool rl);
    PetscInt computeDev_exact(PetscReal a, PetscBool rl);
    PetscReal computeRadius(Mat &A);
    void countInterval(PetscReal a, PetscReal b, PetscInt *count);

public:

    eigen_mm();
    ~eigen_mm();

    int init(Mat &K_in, Mat &M_in, SolverOptions &opt);
    int solve(Mat *V_out, Vec *lambda_out);

    void exactEigenvalues_square_neumann(int Ne, 
        std::vector<PetscReal> &lambda, 
        std::vector<PetscReal> &eta1, 
        std::vector<PetscReal> &eta2);
    void exactEigenvalues_cube_neumann(int Ne, 
        std::vector<PetscReal> &lambda, 
        std::vector<PetscReal> &eta1, 
        std::vector<PetscReal> &eta2,
        std::vector<PetscReal> &eta3);

};

#endif //eigen_mm_H