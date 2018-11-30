cmake_minimum_required(VERSION 3.04)
project(slepc006)

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
        ${PETSC_DIR}/arch-linux2-c-debug/include
)

# -----------------------------------------------------------------------
# SLEPc

#set(SLEPC_DIR "~/softwares/slepc-3.8.3")
set(SLEPC_LIB ${SLEPC_DIR}/${PETSC_ARCH}/lib/libslepc.so)
include_directories(
        ${SLEPC_DIR}/include
        ${SLEPC_DIR}/arch-linux2-c-debug/include
)

# -----------------------------------------------------------------------

add_executable(slepc006 main.cpp eigen_maj.h)

# -----------------------------------------------------------------------
# MPI - part 2/2

if(MPI_COMPILE_FLAGS)
    set_target_properties(slepc006 PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(slepc006 PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# -----------------------------------------------------------------------

target_link_libraries(slepc006 ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB})
