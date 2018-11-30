# eigenSolver

This eigensolver uses PETSc and SLEPc. The following variables should be set:\
**PETSC_DIR\
PETSC_ARCH\
SLEPC_DIR**

To solve a generalized eigenvalue problem, PETSc should be installed with mumps.\
Also, to use the view functions "x" option should be passed to PETSc.\
To do that, when installing PETSc pass the following parameters:\
./configure --with-cxx-dialect=C++11 --download-mumps  --with-x

---

To see an example, check main.cpp.

---

solve() function computes the eigenvalues and eigenvectors.\
solve(int nev, int ncv, int mpd, bool verbose):\
**nev**: number of eigenvalues requested.\
**verbose**: pass "true" to print information.\
**ncv** and **mpd**: From SLEPc documentation:

 The parameters ncv and mpd are intimately related, so that the user is advised to set one of them at most.\
 Normal usage is that (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev);\
 and (b) in cases where nev is large, the user sets mpd.\
 The value of ncv should always be between nev and (nev+mpd), typically ncv=nev+mpd.\
 If nev is not too large, mpd=nev is a reasonable choice, otherwise a smaller value should be used.
 
 If solve() is called without passing aany argument, it will compute **all** the eigenvalues, without printing any information. And, ncv = 2 * nev and mpd = nev.

---
