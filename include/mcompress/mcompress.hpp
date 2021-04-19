#include <mpi.h>
#include <mkl.h>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <algorithm>
#include <chrono>

using std::cout;
using std::endl;
typedef std::chrono::high_resolution_clock Clock;
const double NS2MS = 0.000001;

using BFInt = long;
using PetscInt = int;
#define MPI_BFINT MPI_LONG
#include <bftypes.hpp>
#include <bfutils.hpp>
#include <bfdecode.hpp>
#include <bfencode.hpp>