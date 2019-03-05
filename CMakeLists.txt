cmake_minimum_required(VERSION 3.04)
project(EigenMM)

set(CMAKE_CXX_STANDARD 11)

# -----------------------------------------------------------------------
# MPI

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# -----------------------------------------------------------------------
# PETSc

#Majid
set(PETSC_DIR_EigenMM "~/softwares/petsc-3.8.0")
set(PETSC_ARCH_EigenMM "arch-linux2-c-debug")
#set(PETSC_DIR_EigenMM "~/Software/petsc-3.8.4")
#set(PETSC_ARCH_EigenMM "arch-linux2-c-opt")

set(PETSC_LIB_EigenMM ${PETSC_DIR_EigenMM}/${PETSC_ARCH_EigenMM}/lib/libpetsc.so)
include_directories(
        ${PETSC_DIR_EigenMM}/include
        ${PETSC_DIR_EigenMM}/${PETSC_ARCH_EigenMM}/include
)

# -----------------------------------------------------------------------
# SLEPc

# paste the SLEPC_DIR address here:
#set(SLEPC_DIR /home/majidrp/Projects/nektarpp_eigensolver/library/MultiRegions/eigenSolver/slepc-3.8.3)

set(SLEPC_DIR "~/softwares/slepc-3.8.3")
#set(SLEPC_DIR "~/Software/slepc-3.8.3")
set(SLEPC_LIB ${SLEPC_DIR}/${PETSC_ARCH_EigenMM}/lib/libslepc.so)
include_directories(
        ${SLEPC_DIR}/include
        ${SLEPC_DIR}/${PETSC_ARCH_EigenMM}/include
)

# -----------------------------------------------------------------------

include_directories(include)
set(SOURCE_FILES ${SOURCE_FILES}
        src/eigen_mm.cpp
        src/aux_functions.cpp)
set(HEADER_FILES ${HEADER_FILES}
        include/eigen_mm.h
        )

# -----------------------------------------------------------------------

add_library(EigenMM SHARED ${SOURCE_FILES})
target_link_libraries(EigenMM ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB_EigenMM})
install(FILES ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_DIR}/include)

# -----------------------------------------------------------------------
# tests

#add_executable(test1 src/test_fractional1.cpp ${SOURCE_FILES} ${HEADER_FILES})
#target_link_libraries(test1 ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB_EigenMM})

#add_executable(test1_info src/test_fractional1_info.cpp ${SOURCE_FILES} ${HEADER_FILES})
#target_link_libraries(test1_info ${MPI_LIBRARIES} ${SLEPC_LIB} ${PETSC_LIB_EigenMM})

