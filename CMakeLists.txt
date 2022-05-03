cmake_minimum_required(VERSION 3.04)
#project(EigenMM)

set(CMAKE_CXX_STANDARD 11)

# Set up RPATH
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${LIB_DIR}")
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
        "${CMAKE_INSTALL_PREFIX}/${LIB_DIR}" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_DIR}")
ENDIF("${isSystemDir}" STREQUAL "-1")

# -----------------------------------------------------------------------
# MPI

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# -----------------------------------------------------------------------
# Main Files

set(EIGENMM_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(EIGENMM_INCLUDES ${EIGENMM_DIR}/include/eigenmm)
INCLUDE_DIRECTORIES(${PETSC_INCLUDES})
INCLUDE_DIRECTORIES(${EIGENMM_INCLUDES})

set(EIGENMM_SOURCE_FILES ${EIGENMM_DIR}/src/eigenmm/eigen_mm.cpp)
set(EIGENMM_HEADER_FILES ${EIGENMM_DIR}/include/eigenmm/eigen_mm.h)

add_library(eigenmm SHARED ${EIGENMM_SOURCE_FILES})
target_link_libraries(eigenmm ${MPI_LIBRARIES} ${SLEPC_LIBRARIES} ${PETSC_LIBRARIES})

install(FILES ${EIGENMM_HEADER_FILES} DESTINATION ${TPDIST}/include)
install(TARGETS eigenmm DESTINATION ${TPDIST}/lib)

# If this cmake file is called inside Nektar++, "TPDIST" won't be NULL.
IF(${TPDIST})
    configure_file(${EIGENMM_HEADER_FILES} ${TPDIST}/include)
ENDIF(${TPDIST})
