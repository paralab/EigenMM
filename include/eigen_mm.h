#ifndef eigen_mm_H
#define eigen_mm_H

#include <petsc.h>
#include <slepceps.h>
//#include </home/majidrp/Software/petsc-3.8.4/include/petsc.h> // todo: fix this
//#include </home/majidrp/Projects/nektarpp_eigensolver/library/MultiRegions/EigenMM/slepc-3.8.3/include/slepceps.h> // todo: fix this

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include "mpi.h"

#define _DEBUG1_

template<class T>
int print_vector(const std::vector<T> &v, int ran, const std::string &name, MPI_Comm comm);


class eigen_mm{
    // Generalized eigenvalue problem:
    // A v = \lambda B v

private:

    MPI_Comm comm = MPI_COMM_WORLD;
    Mat A = nullptr;
    Mat B = nullptr; // used for generalized eigenvalue problem
    unsigned int eig_num = 0; // number of computed eigenvaleus.

public:

    bool store_eigenpairs = true; // set this to true to store eigenpairs in the wrapper class.
    std::vector<double> eig_val_real; // real part of eigenvalues. the same on all the processors
    std::vector<double> eig_val_imag; // imaginary part of eigenvalues. the same on all the processors
    std::vector< std::vector<double> > eig_vec_real; // real part of eigenvectors. local part of the vector on each processor
    std::vector< std::vector<double> > eig_vec_imag; // imaginary part of eigenvectors. local part of the vector on each processor

    eigen_mm();
    ~eigen_mm();

    int init(unsigned int mat_size, MPI_Comm com);
    int setA(unsigned int i, unsigned int j, double v);
    int setB(unsigned int i, unsigned int j, double v);
    int assemble();
    int solve(int nev = 0, int ncv = 0, int mpd = 0, bool verbose = false);
    int solve_interval(int nev = 0, int ncv = 0, int mpd = 0, bool verbose = false);

    void print_eig_val_real();
    void print_eig_val_imag();
    void print_eig_val();
    void print_eig_vec_real(int ran);
    void print_eig_vec_imag(int ran);
    void print_eig_vec(int ran);

    int viewA();
    int viewB();


    int get_eig_num();
    double* get_eig_val_real();
    double* get_eig_val_imag();
    double* get_eig_vec_real(int i);
    double* get_eig_vec_imag(int i);
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

#endif //eigen_mm_H
