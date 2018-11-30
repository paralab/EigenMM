# eigenSolver

This eigensolver uses PETSc and SLEPc. The following variables should be set:\
PETSC_DIR\
PETSC_ARCH\
SLEPC_DIR

To solve a generalized eigenvalue problem, PETSc should be installed with mumps.\
Also, to use the view functions "x" option should be passed to PETSc. To do that, when installing PETSc pass the following parameters:\
./configure --with-cxx-dialect=C++11 --download-mumps  --with-x

solve() function computes the eigenvalues and eigenvectors. 
