cmake_minimum_required(VERSION 3.04)
project(EigenMMAndMCompress)

set(CMAKE_CXX_STANDARD 11)

# -----------------------------------------------------------------------
# MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
# -----------------------------------------------------------------------

# PETSc
set(PETSC_ROOT "" CACHE STRING "Path to PETSc root directory.")
set(PETSC_ARCH "arch-linux2-c-opt" CACHE STRING "PETSc architecture.")
set(PETSC_LIB ${PETSC_ROOT}/${PETSC_ARCH}/lib/libpetsc.so)
include_directories(
        ${PETSC_ROOT}/include
        ${PETSC_ROOT}/${PETSC_ARCH}/include
)
# -----------------------------------------------------------------------

# SLEPc
set(SLEPC_ROOT  "" CACHE STRING "Path to SLEPc root directory.")
set(SLEPC_LIB ${SLEPC_ROOT}/${PETSC_ARCH}/lib/libslepc.so)
include_directories(
        ${SLEPC_ROOT}/include
        ${SLEPC_ROOT}/${PETSC_ARCH}/include
)
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# TinyXML
set(TINYXML_INCLUDE "" CACHE STRING "Path to TinyXML include directory.")
set(TINYXML_LIB "" CACHE STRING "Path to TinyXML library.")
include_directories(${TINYXML_INCLUDE})
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# MKL
set(MKL_INCLUDE_DIRS "" CACHE STRING "Path to MKL include directory.")
set(MKL_LIBRARIES "" CACHE STRING "Path to MKL libraries directory.")
include_directories(${MKL_INCLUDE_DIRS})
link_directories(${MKL_LIBRARIES})
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# EigenMM
include_directories(include/eigenmm/)
add_library(EigenMM SHARED src/eigenmm/eigen_mm.cpp)
target_link_libraries(EigenMM ${MPI_LIBRARIES} ${TINYXML_LIB} ${SLEPC_LIB} ${PETSC_LIB})
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# MCompress
include_directories(include/mcompress/)
# -----------------------------------------------------------------------

add_executable(EigenMMSolve src/eigenmm/eigen_mm_solve.cpp include/eigenmm/eigen_mm.h)
target_link_libraries(EigenMMSolve ${MPI_LIBRARIES} ${TINYXML_LIB} ${SLEPC_LIB} ${PETSC_LIB} EigenMM)

add_executable(MCompressEncode src/mcompress/encode.cpp include/mcompress/mcompress.hpp)
target_link_libraries(MCompressEncode ${MPI_LIBRARIES} mkl_intel_lp64 mkl_sequential mkl_core)

add_executable(MCompressApply src/mcompress/apply.cpp include/mcompress/mcompress.hpp)
target_link_libraries(MCompressApply ${MPI_LIBRARIES} mkl_intel_lp64 mkl_sequential mkl_core)

add_executable(MCompressDenseMatvec src/mcompress/dense_matvec.cpp include/mcompress/mcompress.hpp)
target_link_libraries(MCompressDenseMatvec ${MPI_LIBRARIES} ${PETSC_LIB} mkl_intel_lp64 mkl_sequential mkl_core)