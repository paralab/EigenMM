#!/bin/bash

module load mumps/5.0.2
module load petsc/3.8.3

git clone https://github.com/paralab/eigenSolver.git
cd eigenSolver

tar -xzf slepc-3.8.3.tar.gz
cd slepc-3.8.3
export SLEPC_DIR='pwd'
make
