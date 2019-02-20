#!/bin/bash

# set address for the nektarpp_eigensolver source directory
export NEK_EIG=/home/majidrp/Projects/nektarpp_eigensolver/

if [ -z "${NEK_EIG}" ]; then
    echo "set NEK_EIG to nektarpp_eigensolver source directory."
fi

export PETSC_DIR=${NEK_EIG}/build/ThirdParty/petsc-3.7.7
export PETSC_ARCH=c-opt

if [ ! -d "slepc-3.7.4" ]; then
    tar -xzf slepc-3.7.4.tar.gz
fi
cd slepc-3.7.4
export SLEPC_DIR=`pwd`
./configure
make

printf "\ncopy the following line inside CMakeLists.txt after the line that reads: paste the SLEPC_DIR address here:"
printf "\nset(SLEPC_DIR `pwd`)\n"
