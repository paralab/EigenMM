#!/bin/bash

# set PETSc environment variables
export PETSC_DIR=
export PETSC_ARCH=

if [ -z "${PETSC_DIR}" ]; then
    echo "set PETSC_DIR to PETSc source directory."
fi
if [ -z "${PETSC_ARCH}" ]; then
    echo "set PETSC_ARCH."
fi

if [ ! -d "slepc-3.8.3" ]; then
    tar -xzf slepc-3.8.3.tar.gz
fi
cd slepc-3.8.3
export SLEPC_DIR=`pwd`
./configure
make

printf "\ncopy the following line inside CMakeLists.txt after the line that reads: paste the SLEPC_DIR address here:"
printf "\nset(SLEPC_DIR `pwd`)\n"
