#ifndef EIGENMM_AUX_FUNCTIONS_H
#define EIGENMM_AUX_FUNCTIONS_H

#include <vector>
#include <string>
#include <iostream>
#include "mpi.h"

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

double print_time(double t_start, double t_end, std::string function_name, MPI_Comm comm);

#endif //EIGENMM_AUX_FUNCTIONS_H
