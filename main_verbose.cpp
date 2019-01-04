#include "eigen_maj.h"
#include <slepceps.h>
#include <iostream>
#include <vector>
#include <ctime>
#include "mpi.h"

#define _DEBUG1_

typedef unsigned int index_t;
typedef double value_t;

int laplacian3D(eigen_maj &eigSolver, int mx, int my, int mz);
int laplacian3D_randomized(eigen_maj &eigSolver, int mx, int my, int mz);
double print_time(double t_start, double t_end, std::string function_name, MPI_Comm comm);

#undef __FUNCT__
#define __FUNCT__ "main"

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(rank == 0){
        printf("\nUsage:   mpirun -np <#procs> ./eigenSolver <x-axis size> <y-axis size> <z-axis size> \n");
        printf("example: mpirun -np 3 ./eigenSolver 3 3 3 \n\n");
    }

    int mx(std::stoi(argv[1]));
//    int my(std::stoi(argv[2]));
//    int mz(std::stoi(argv[3]));
    int my = mx;
    int mz = mx;
    auto matrix_sz = unsigned(mx * my * mz);

    eigen_maj eigSolver;                            // define
    eigSolver.init(matrix_sz);                      // initialize. matrix_sz: matrix size (number of rows)

    laplacian3D_randomized(eigSolver, mx, my, mz);  // set matrix A: an example how to use eigSolver.setA(row, col, val) to set values.
    laplacian3D(eigSolver, mx, my, mz);             // set matrix B: an example how to use eigSolver.setB(row, col, val) to set values.

#ifdef _DEBUG1_
    double t1, t2;
    MPI_Barrier(comm);
    t1 = MPI_Wtime();
#endif

    eigSolver.assemble();                           // assemble

#ifdef _DEBUG1_
    t2 = MPI_Wtime();
    print_time(t1, t2, "assemble", comm);
#endif

//    eigSolver.viewA();                            // view matrix A
//    eigSolver.viewB();                            // view matrix B

#ifdef _DEBUG1_
    MPI_Barrier(comm);
    t1 = MPI_Wtime();
#endif

    int nev = matrix_sz;
    int mpd(std::stoi(argv[2]));
//    int ncv(std::stoi(argv[3]));
    int ncv = nev + mpd;
    if(rank==0) printf("size = %d, nev = %d, ncv = %d, mpd = %d\n", matrix_sz, nev, ncv, mpd);

    eigSolver.solve();                              // find eigenvalues and eigenvectors
//    eigSolver.solve(nev, ncv, mpd, false);          // find eigenvalues and eigenvectors

#ifdef _DEBUG1_
    t2 = MPI_Wtime();
    print_time(t1, t2, "solve", comm);
    MPI_Barrier(comm);
#endif

//    eigSolver.print_eig_val();                    // print eigenvalues (complex form)
//    eigSolver.print_eig_val_real();               // print eigenvalues (real part)
//    eigSolver.print_eig_val_imag();               // print eigenvalues (imaginary part)

//    eigSolver.print_eig_vec(-1);                  // print eigenvectors (complex form)
//    eigSolver.print_eig_vec_real(-1);             // print eigenvectors (real part)
//    eigSolver.print_eig_vec_imag(-1);             // print eigenvectors (imaginary part)


    // this part shows how to access eigenvalues and eigenvectors
    // ----------------------------------------------------------
    /*
    int eig_num = eigSolver.get_eig_num();

//    MPI_Barrier(comm);
//    if(rank==0) printf("number of eigenvalues computed: %u\n", eig_num);
//    MPI_Barrier(comm);

    double *eig_val_real = eigSolver.get_eig_val_real();
    double *eig_val_imag = eigSolver.get_eig_val_imag();

    MPI_Barrier(comm);
    if(rank == nprocs-1) {
        printf("\neigenvalues on processor %d: \n", rank);
        for (int i = 0; i < eig_num; i++) {
            printf("%d \t %f + %f i\n", i, eig_val_real[i], eig_val_imag[i]);
        }
    }
    MPI_Barrier(comm);

    int eig_vec_i = 1;
    double *eig_vec_i_real;
    eig_vec_i_real = eigSolver.get_eig_vec_real(1);
    double *eig_vec_i_imag;
    eig_vec_i_imag = eigSolver.get_eig_vec_imag(1);

    if(rank == 0) {
        printf("\neigenvector %d on processor %d: \n", eig_vec_i, rank);
        for (int i = 0; i < eig_num; i++) {
            printf("%d \t %f + %f i\n", i, eig_vec_i_real[i], eig_vec_i_imag[i]);
        }
    }
*/

    return 0;
}


int laplacian3D(eigen_maj &eigSolver, int mx, int my, int mz){

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int     i,j,k,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    value_t v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    index_t col_index[7];
    index_t node;

    Hx      = 1.0 / (value_t)(mx);
    Hy      = 1.0 / (value_t)(my);
    Hz      = 1.0 / (value_t)(mz);

    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;
    zm = (int)floor(mz / nprocs);
    zs = rank * zm;
    if(rank == nprocs - 1)
        zm = mz - ( (nprocs - 1) * zm);

    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                node = mx * my * k + mx * j + i;

                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
                    num = 0; numi=0; numj=0; numk=0;
                    if (k!=0) {
                        v[num]     = -HxHydHz;
                        col_index[num] = node - (mx * my);
                        num++; numk++;
                    }
                    if (j!=0) {
                        v[num]     = -HxHzdHy;
                        col_index[num] = node - mx;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HyHzdHx;
                        col_index[num] = node - 1;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HyHzdHx;
                        col_index[num] = node + 1;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxHzdHy;
                        col_index[num] = node + mx;
                        num++; numj++;
                    }
                    if (k!=mz-1) {
                        v[num]     = -HxHydHz;
                        col_index[num] = node + (mx * my);
                        num++; numk++;
                    }
                    v[num]     = (value_t)(numk)*HxHydHz + (value_t)(numj)*HxHzdHy + (value_t)(numi)*HyHzdHx;
                    col_index[num] = node;
                    num++;
                    for(int l = 0; l < num; l++){
                        eigSolver.setB(node, col_index[l], v[l]);
                    }

                } else {

                    v[0] = -HxHydHz;
                    col_index[0] = node - (mx * my);
                    eigSolver.setB(node, col_index[0], v[0]);

                    v[1] = -HxHzdHy;
                    col_index[1] = node - mx;
                    eigSolver.setB(node, col_index[1], v[1]);

                    v[2] = -HyHzdHx;
                    col_index[2] = node - 1;
                    eigSolver.setB(node, col_index[2], v[2]);

                    v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
                    col_index[3] = node;
                    eigSolver.setB(node, col_index[3], v[3]);

                    v[4] = -HyHzdHx;
                    col_index[4] = node + 1;
                    eigSolver.setB(node, col_index[4], v[4]);

                    v[5] = -HxHzdHy;
                    col_index[5] = node + mx;
                    eigSolver.setB(node, col_index[5], v[5]);

                    v[6] = -HxHydHz;
                    col_index[6] = node + (mx * my);
                    eigSolver.setB(node, col_index[6], v[6]);

                }
            }
        }
    }

    return 0;
}


int laplacian3D_randomized(eigen_maj &eigSolver, int mx, int my, int mz){

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    srand(time(NULL));

    int     i,j,k,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    value_t v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    index_t col_index[7];
    index_t node;

    Hx      = 1.0 / (value_t)(mx);
    Hy      = 1.0 / (value_t)(my);
    Hz      = 1.0 / (value_t)(mz);

    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;
    zm = (int)floor(mz / nprocs);
    zs = rank * zm;
    if(rank == nprocs - 1)
        zm = mz - ( (nprocs - 1) * zm);

    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                node = mx * my * k + mx * j + i;

                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
                    num = 0; numi=0; numj=0; numk=0;
                    if (k!=0) {
                        v[num]     = -HxHydHz;
                        col_index[num] = node - (mx * my);
                        num++; numk++;
                    }
                    if (j!=0) {
                        v[num]     = -HxHzdHy;
                        col_index[num] = node - mx;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HyHzdHx;
                        col_index[num] = node - 1;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HyHzdHx;
                        col_index[num] = node + 1;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxHzdHy;
                        col_index[num] = node + mx;
                        num++; numj++;
                    }
                    if (k!=mz-1) {
                        v[num]     = -HxHydHz;
                        col_index[num] = node + (mx * my);
                        num++; numk++;
                    }
                    v[num]     = (value_t)(numk)*HxHydHz + (value_t)(numj)*HxHzdHy + (value_t)(numi)*HyHzdHx;
                    col_index[num] = node;
                    num++;
                    for(int l = 0; l < num; l++){
                        eigSolver.setA(node, col_index[l], (float(rand() %10) + 1)/10 * v[l]);
                    }

                } else {

                    v[0] = -HxHydHz;
                    col_index[0] = node - (mx * my);
                    eigSolver.setA(node, col_index[0], (float(rand() %10) + 1)/10 * v[0]);

                    v[1] = -HxHzdHy;
                    col_index[1] = node - mx;
                    eigSolver.setA(node, col_index[1], (float(rand() %10) + 1)/10 * v[1]);

                    v[2] = -HyHzdHx;
                    col_index[2] = node - 1;
                    eigSolver.setA(node, col_index[2], (float(rand() %10) + 1)/10 * v[2]);

                    v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
                    col_index[3] = node;
                    eigSolver.setA(node, col_index[3], (float(rand() %10) + 1)/10 * v[3]);

                    v[4] = -HyHzdHx;
                    col_index[4] = node + 1;
                    eigSolver.setA(node, col_index[4], (float(rand() %10) + 1)/10 * v[4]);

                    v[5] = -HxHzdHy;
                    col_index[5] = node + mx;
                    eigSolver.setA(node, col_index[5], (float(rand() %10) + 1)/10 * v[5]);

                    v[6] = -HxHydHz;
                    col_index[6] = node + (mx * my);
                    eigSolver.setA(node, col_index[6], (float(rand() %10) + 1)/10 * v[6]);

                }
            }
        }
    }

    return 0;
}


double print_time(double t_start, double t_end, std::string function_name, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double min, max, average;
    double t_dif = t_end - t_start;

    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\nmin: " << min << "\nave: " << average << "\nmax: " << max << std::endl << std::endl;

    return average;
}