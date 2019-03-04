#!/bin/bash

module load mumps/5.0.2
module load petsc/3.8.3

export PETSC_ARCH=

if [ ! -d "slepc-3.8.3" ]; then
    tar -xzf slepc-3.8.3.tar.gz
fi
cd slepc-3.8.3
export SLEPC_DIR=`pwd`
./configure
make

printf "\ncopy the following line inside CMakeLists.txt after the line that reads: paste the SLEPC_DIR address here:"
printf "\nset(SLEPC_DIR `pwd`)\n"
