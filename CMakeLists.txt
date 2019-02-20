cmake_minimum_required(VERSION 3.04)
project(eigenSolver)

set(CMAKE_CXX_STANDARD 11)

# -----------------------------------------------------------------------
# MPI

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# -----------------------------------------------------------------------
# PETSc

#Majid
set(PETSC_DIR "/home/majidrp/Projects/nektarpp_saena/build/ThirdParty/petsc-3.7.7")
set(PETSC_ARCH "c-opt")
set(PETSC_LIB ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
include_directories(
        ${PETSC_DIR}/include
        ${PETSC_DIR}/${PETSC_ARCH}/include
)

# -----------------------------------------------------------------------
# SLEPc

# paste the SLEPC_DIR address here:
set(SLEPC_DIR /home/majidrp/Projects/nektarpp_eigensolver/library/MultiRegions/eigenSolver/slepc-3.7.4)

#set(SLEPC_DIR "~/Software/slepc-3.8.3")
set(SLEPC_LIB ${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.so)
include_directories(
        ${SLEPC_DIR}/include
        ${SLEPC_DIR}/${PETSC_ARCH}/include
)

# -----------------------------------------------------------------------

#add_executable(eigenSolver main.cpp eigen_mm.h)
#target_link_libraries(eigenSolver ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB})

# this one prints information
add_executable(eigenSolver main_verbose.cpp eigen_mm.h)
target_link_libraries(eigenSolver ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB})
