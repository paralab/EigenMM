# EigenMM

This eigensolver uses PETSc and SLEPc. The following variables should be set:\
**PETSC_DIR\
PETSC_ARCH\
SLEPC_DIR**

To solve a generalized eigenvalue problem, PETSc should be installed with mumps.\
Also, to use the "view" functions, the "x" option should be passed to PETSc.\
To do that, when installing PETSc pass the following parameters:\
./configure --with-cxx-dialect=C++11 --download-mumps  --with-x

---

To see an example, check test_fractional2.cpp in folder src.

---

To use this solver include the header file **eigen_mm.h**.

---
