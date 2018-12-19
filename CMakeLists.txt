cmake_minimum_required(VERSION 3.04)
project(eigenSolver)

set(CMAKE_CXX_STANDARD 11)

# -----------------------------------------------------------------------
# MPI - part 1/2

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# -----------------------------------------------------------------------
# PETSc

#set(PETSC_DIR "~/softwares/petsc-3.8.4")
#set(PETSC_ARCH "arch-linux2-c-debug")
set(PETSC_LIB ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
include_directories(
        ${PETSC_DIR}/include
        ${PETSC_DIR}/${PETSC_ARCH}/include
)

# -----------------------------------------------------------------------
# SLEPc

# paste the SLEPC_DIR address here:


#set(SLEPC_DIR "~/softwares/slepc-3.8.3")
set(SLEPC_LIB ${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.so)
include_directories(
        ${SLEPC_DIR}/include
        ${SLEPC_DIR}/${PETSC_ARCH}/include
)

# -----------------------------------------------------------------------

add_executable(eigenSolver main.cpp eigen_maj.h)
target_link_libraries(eigenSolver ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB})
