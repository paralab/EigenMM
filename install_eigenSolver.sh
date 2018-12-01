#!/bin/bash

module load mumps/5.0.2
module load petsc/3.8.3

tar -xzf slepc-3.8.3.tar.gz
cd slepc-3.8.3
export SLEPC_DIR=`pwd`
./configure
make

printf "\ncopy the following line inside CMakeLists.txt after the line that reads: paste the SLEPC_DIR address here:"
printf "set(SLEPC_DIR `pwd`)\n"
