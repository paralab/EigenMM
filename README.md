# eigenSolver

This eigensolver uses PETSc and SLEPc. The following variables should be set:\
PETSC_DIR\
PETSC_ARCH\
PETSC_LIB

PETSc should be installed with mumps. To use the view functions "x" should be used. To do that:\
./configure --with-cxx-dialect=C++11 --download-mumps  --with-x
