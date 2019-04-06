#!/bin/bash

# set PETSc environment variables
echo "inside the script, PETSC_DIR:"
echo ${PETSC_DIR}

if [ ! -d "slepc-3.7.4" ]; then
    tar -xzf slepc-3.7.4.tar.gz
fi
cd slepc-3.7.4
export SLEPC_DIR=`pwd`
./configure
make

#echo "inside the script, SLEPC_DIR:"
#echo ${SLEPC_DIR}
#printf "\ncopy the following line inside CMakeLists.txt after the line that reads: paste the SLEPC_DIR address here:"
#printf "\nset(SLEPC_DIR `pwd`)\n"
