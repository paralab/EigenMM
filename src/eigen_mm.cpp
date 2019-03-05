#include "eigen_mm.h"

eigen_mm::eigen_mm() = default;

eigen_mm::~eigen_mm(){
    MatDestroy(&A);
    MatDestroy(&B);
    SlepcFinalize();
};


int eigen_mm::init(const unsigned int mat_size, MPI_Comm com){

    SlepcInitialize(0, nullptr, nullptr, nullptr);

    comm = com;

    MatCreate(comm, &A);
    MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, mat_size, mat_size);
    MatSetFromOptions(A);
    MatSetUp(A);

    MatCreate(comm, &B);
    MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, mat_size, mat_size);
    MatSetFromOptions(B);
    MatSetUp(B);

    return 0;
}


int eigen_mm::setA(const unsigned int i, const unsigned int j, const double v){
    MatSetValue(A, i, j, v, ADD_VALUES);
    return 0;
}

int eigen_mm::setB(const unsigned int i, const unsigned int j, const double v){
    MatSetValue(B, i, j, v, ADD_VALUES);
    return 0;
}


int eigen_mm::assemble(){

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,   MAT_FINAL_ASSEMBLY);

    return 0;
}


// old version
// int eigen_mm::solve(int nev = 0, int ncv = 0, int mpd = 0, bool verbose = false)
/*
    int solve(int nev = 0, int ncv = 0, int mpd = 0, bool verbose = false){

        // nev: number of requested eigenvalues. enter 0 to request all eigenvalues.
        // verbose: pass "true" to print information.
        // Saves the real part of eigenvalues in eig_val_real and their imaginary part in eig_val_imag.
        // And, saves the real part of eigenvectors in eig_vec_real and their imaginary part in eig_vec_imag.

        EPS            eps;
        ST             st;
        KSP            ksp;
        EPSType        type;
        PetscReal      tol;
        Vec            xr, xi, *Iv, *Cv;
        PetscInt       maxit, i, its, lits, nconv, nini=0, ncon=0;
        PetscViewer    viewer;
        PetscBool      ishermitian;
        PetscErrorCode ierr;

        // default:
//        bool verbose1 = verbose;
//        bool verbose2 = verbose;

        bool verbose1 = true;
        bool verbose2 = false;

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //              Create the eigensolver and set various options
        //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        //   Create eigensolver context

        ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

        // Set operators. In this case, it is a generalized eigenvalue problem

        ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);
//        ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr); // for standard
        ierr = EPSSetProblemType(eps,EPS_GHIEP);CHKERRQ(ierr);
        // EPS_HEP: Hermitian eigenvalue problem.
        // for generalized eigenproblem both A and B should be Hermitian (symmetric for real matrices) and B should be
        // positive (semi-)definite to use ESP_HEP. If B is not positive (semi-)definite then the problem cannot be considered
        //Hermitian but symmetry can still be exploited to some extent in some solvers (problem type EPS_GHIEP)

        double tol1 = 1e-8;
        int max_iter = 1000;
        EPSSetTolerances(eps,tol1, max_iter);

        // If the user provided initial guesses or constraints, pass them here

        ierr = EPSSetInitialSpace(eps,nini,Iv);CHKERRQ(ierr);
        ierr = EPSSetDeflationSpace(eps,ncon,Cv);CHKERRQ(ierr);

        // save matrix size
        int mat_size;
        MatGetSize(A, &mat_size, NULL);
//        PetscPrintf(PETSC_COMM_WORLD," mat_size = %D\n", mat_size);

        // set requested number of eigenvalues here:
        // from documentation:
        // EPSSetDimensions(EPS eps, PetscInt nev, PetscInt ncv, PetscInt mpd);
        // The parameters ncv and mpd are intimately related, so that the user is advised to set one of them at most.
        // Normal usage is that (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev);
        // and (b) in cases where nev is large, the user sets mpd.
        // The value of ncv should always be between nev and (nev+mpd), typically ncv=nev+mpd.
        // If nev is not too large, mpd=nev is a reasonable choice, otherwise a smaller value should be used.

        // ncv: number of column vectors (i.e., the dimension of the subspace with which the eigensolver works).
        // This usually improves the convergence behavior at the expense of larger memory requirements.

        if(nev == 0) nev = mat_size; // if nev is not set, request all eigenvalues.
        if(ncv == 0) ncv = 2 * nev;  // if ncv is not set, set it to default.
        if(mpd == 0) mpd = nev;      // if mpd is not set, set it to default.
        EPSSetDimensions(eps, nev, ncv, mpd);

        // Set solver parameters at runtime

        ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //                    Solve the eigensystem
        //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        ierr = EPSSolve(eps);CHKERRQ(ierr);


        // Optional: Get some information from the solver and display it

        if(verbose1){
            ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
            ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
            ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
            ierr = KSPGetTotalIterations(ksp,&lits);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %D\n",lits);CHKERRQ(ierr);
            ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
            ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
            ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
            ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //                  Display solution and clean up
        //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // Show detailed info if verbose is true.

        if(verbose2){
            ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
            ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }


        // store eigenvalues and eigenvectors in the class.

        ierr = MatCreateVecs(A,NULL,&xr);CHKERRQ(ierr);
        ierr = MatCreateVecs(A,NULL,&xi);CHKERRQ(ierr);

        int nlocal;
        VecGetLocalSize(xr, &nlocal);

        ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
        eig_val_real.resize(nconv);
        eig_val_imag.resize(nconv);
        eig_vec_real.resize(nconv, std::vector<double> (nlocal));
        eig_vec_imag.resize(nconv, std::vector<double> (nlocal));

        eig_num = nconv;

//        ierr = EPSIsHermitian(eps,&ishermitian);CHKERRQ(ierr);

        double* vec;
        for (i = 0; i < nconv; i++) {
            EPSGetEigenpair(eps, i, &eig_val_real[i], &eig_val_imag[i], xr, xi);
            VecGetArray(xr, &vec);
            memcpy(&eig_vec_real[i][0], &vec[0], nlocal * sizeof(double));
            VecGetArray(xi, &vec);
            memcpy(&eig_vec_imag[i][0], &vec[0], nlocal * sizeof(double));
        }


        // Free work space

        ierr = EPSDestroy(&eps);CHKERRQ(ierr);
        ierr = VecDestroy(&xr);CHKERRQ(ierr);
        ierr = VecDestroy(&xi);CHKERRQ(ierr);
        if (nini > 0) {
            ierr = VecDestroyVecs(nini,&Iv);CHKERRQ(ierr);
        }
        if (ncon > 0) {
            ierr = VecDestroyVecs(ncon,&Cv);CHKERRQ(ierr);
        }

        return 0;
    }
*/

int eigen_mm::solve(int nev, int ncv, int mpd, bool verbose){

    // nev: number of requested eigenvalues. enter 0 to request all eigenvalues.
    // verbose: pass "true" to print information.
    // Saves the real part of eigenvalues in eig_val_real and their imaginary part in eig_val_imag.
    // And, saves the real part of eigenvectors in eig_vec_real and their imaginary part in eig_vec_imag.

    EPS            eps;
    ST             st;
    KSP            ksp;
    PC             pc;
    EPSType        type;
    PetscReal      tol;
    Vec            xr=nullptr, xi=nullptr, *Iv, *Cv;
    PetscInt       maxit, i, its, lits, nconv, nini=0, ncon=0;
//        PetscViewer    viewer;
//        PetscBool      ishermitian;
    PetscErrorCode ierr;

    // default:
//        bool verbose1 = verbose;
//        bool verbose2 = verbose;

    bool verbose1 = false;
    bool verbose2 = false;

    // set command-line parameters here
    // Hardwire several options; can be changed at command line
    char common_options[] = "-mat_mumps_icntl_13 1";
    PetscOptionsInsertString(nullptr, common_options);
//        PetscOptionsInsert(&argc,&argv,PETSC_NULL);
    // PetscOptionsInsert: It "resets" the options database with the command line options.
    // If you did not call this then the InsertString() options would not be overwritten with the command line options.
    // https://lists.mcs.anl.gov/pipermail/petsc-users/2012-May/013464.html

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //              Create the eigensolver and set various options
    //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    //   Create eigensolver context

    ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

    // Set operators. In this case, it is a generalized eigenvalue problem.

//        ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr); // for standard
    ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);

    // set problem type:
    // Hermitian (EPS_HEP),
    // non-Hermitian (EPS_NHEP),
    // generalized Hermitian (EPS_GHEP),
    // generalized non-Hermitian (EPS_GNHEP),
    // generalized non-Hermitian with positive semi-definite B (EPS_PGNHEP),
    // generalized Hermitian-indefinite (EPS_GHIEP).
    // for generalized eigenproblem both A and B should be Hermitian (symmetric for real matrices) and B should be
    // positive (semi-)definite to use ESP_HEP. If B is not positive (semi-)definite then the problem cannot be considered
    // Hermitian but symmetry can still be exploited to some extent in some solvers (problem type EPS_GHIEP)
    // http://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetProblemType.html
    ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);

    double tol1 = 1e-8;
    int max_iter = 1000;
    EPSSetTolerances(eps,tol1, max_iter);

    // If the user provided initial guesses or constraints, pass them here

//        ierr = EPSSetInitialSpace(eps,nini,Iv);CHKERRQ(ierr);
//        ierr = EPSSetDeflationSpace(eps,ncon,Cv);CHKERRQ(ierr);

    ierr = EPSSetWhichEigenpairs(eps,EPS_ALL);CHKERRQ(ierr);
    ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
    ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
    ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
//        ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    // save matrix size
    int mat_size;
    MatGetSize(A, &mat_size, NULL);
//        PetscPrintf(PETSC_COMM_WORLD," mat_size = %D\n", mat_size);

    // set requested number of eigenvalues here:
    // from documentation:
    // EPSSetDimensions(EPS eps, PetscInt nev, PetscInt ncv, PetscInt mpd);
    // The parameters ncv and mpd are intimately related, so that the user is advised to set one of them at most.
    // Normal usage is that (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev);
    // and (b) in cases where nev is large, the user sets mpd.
    // The value of ncv should always be between nev and (nev+mpd), typically ncv=nev+mpd.
    // If nev is not too large, mpd=nev is a reasonable choice, otherwise a smaller value should be used.

    // ncv (number of column vectors): (i.e., the dimension of the subspace with which the eigensolver works).
    // This usually improves the convergence behavior at the expense of larger memory requirements.
    // mpd (maximum projected dimension): (read about the steps on page 45 of the manual)
    // The idea is to bound the size of the projected eigenproblem so that steps 3 and 4 work with a dimension
    // of mpd at most, while steps 1 and 2 still work with a bigger dimension, up to ncv.

    if(nev == 0) nev = mat_size; // if nev is not set, request all eigenvalues.
    if(ncv == 0) ncv = 2 * nev;  // if ncv is not set, set it to default.
    if(mpd == 0) mpd = nev;      // if mpd is not set, set it to default.
    EPSSetDimensions(eps, nev, ncv, mpd);

    // Set solver parameters at runtime

    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                    Solve the eigensystem
    //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ierr = EPSSolve(eps);CHKERRQ(ierr);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Display solving related information
    //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    if(verbose1){
        ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
        ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
        ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
        ierr = KSPGetTotalIterations(ksp,&lits);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %D\n",lits);CHKERRQ(ierr);
        ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
        ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
        ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
        ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Display and save solution
    //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    // Show detailed info if verbose is true.

    if(verbose2){
        ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
        ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    // store eigenvalues and eigenvectors in the class.
    if(store_eigenpairs) {
        ierr = MatCreateVecs(A, NULL, &xr);
        CHKERRQ(ierr);
        ierr = MatCreateVecs(A, NULL, &xi);
        CHKERRQ(ierr);

        int nlocal;
        VecGetLocalSize(xr, &nlocal);

        ierr = EPSGetConverged(eps, &nconv);
        CHKERRQ(ierr);
        eig_val_real.resize(nconv);
        eig_val_imag.resize(nconv);
        eig_vec_real.resize(nconv, std::vector<double>(nlocal));
        eig_vec_imag.resize(nconv, std::vector<double>(nlocal));

        eig_num = nconv;

//            ierr = EPSIsHermitian(eps,&ishermitian);CHKERRQ(ierr);

        double *vec;
        for (i = 0; i < nconv; i++) {
            EPSGetEigenpair(eps, i, &eig_val_real[i], &eig_val_imag[i], xr, xi);
            VecGetArray(xr, &vec);
            memcpy(&eig_vec_real[i][0], &vec[0], nlocal * sizeof(double));
            VecGetArray(xi, &vec);
            memcpy(&eig_vec_imag[i][0], &vec[0], nlocal * sizeof(double));
        }
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                  Free work space
    //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    if(store_eigenpairs){
        ierr = VecDestroy(&xr);CHKERRQ(ierr);
        ierr = VecDestroy(&xi);CHKERRQ(ierr);
    }
    if (nini > 0) {
        ierr = VecDestroyVecs(nini,&Iv);CHKERRQ(ierr);
    }
    if (ncon > 0) {
        ierr = VecDestroyVecs(ncon,&Cv);CHKERRQ(ierr);
    }

    return 0;
}


int eigen_mm::solve_interval(int nev, int ncv, int mpd, bool verbose){

    // nev: number of requested eigenvalues. enter 0 to request all eigenvalues.
    // verbose: pass "true" to print information.
    // Saves the real part of eigenvalues in eig_val_real and their imaginary part in eig_val_imag.
    // And, saves the real part of eigenvectors in eig_vec_real and their imaginary part in eig_vec_imag.

    EPS            eps;
    ST             st;
    KSP            ksp;
    PC             pc;
    EPSType        type;
    PetscReal      tol;
    Vec            xr=nullptr, xi=nullptr, *Iv, *Cv;
    PetscInt       maxit, i, its, lits, nconv, nini=0, ncon=0;
//        PetscViewer    viewer;
//        PetscBool      ishermitian;
    PetscErrorCode ierr;

    // default:
//        bool verbose1 = verbose;
//        bool verbose2 = verbose;

    bool verbose1 = false;
    bool verbose2 = false;

    // set command-line parameters here
    // Hardwire several options; can be changed at command line
    char common_options[] = "-mat_mumps_icntl_13 1";
    PetscOptionsInsertString(nullptr, common_options);
//        PetscOptionsInsert(&argc,&argv,PETSC_NULL);
    // PetscOptionsInsert: It "resets" the options database with the command line options.
    // If you did not call this then the InsertString() options would not be overwritten with the command line options.
    // https://lists.mcs.anl.gov/pipermail/petsc-users/2012-May/013464.html

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //              Create the eigensolver and set various options
    //  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    //   Create eigensolver context

    ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

    // Set operators. In this case, it is a generalized eigenvalue problem.

//        ierr = EPSSetOperators(eps,A,NULL);CHKERRQ(ierr); // for standard
    ierr = EPSSetOperators(eps,A,B);CHKERRQ(ierr);

    // set problem type:
    // Hermitian (EPS_HEP),
    // non-Hermitian (EPS_NHEP),
    // generalized Hermitian (EPS_GHEP),
    // generalized non-Hermitian (EPS_GNHEP),
    // generalized non-Hermitian with positive semi-definite B (EPS_PGNHEP),
    // generalized Hermitian-indefinite (EPS_GHIEP).
    // for generalized eigenproblem both A and B should be Hermitian (symmetric for real matrices) and B should be
    // positive (semi-)definite to use ESP_HEP. If B is not positive (semi-)definite then the problem cannot be considered
    // Hermitian but symmetry can still be exploited to some extent in some solvers (problem type EPS_GHIEP)
    // http://slepc.upv.es/documentation/current/docs/manualpages/EPS/EPSSetProblemType.html
    ierr = EPSSetProblemType(eps,EPS_GHEP);CHKERRQ(ierr);

    double tol1 = 1e-8;
    int max_iter = 1000;
    EPSSetTolerances(eps,tol1, max_iter);

    // If the user provided initial guesses or constraints, pass them here

//        ierr = EPSSetInitialSpace(eps,nini,Iv);CHKERRQ(ierr);
//        ierr = EPSSetDeflationSpace(eps,ncon,Cv);CHKERRQ(ierr);

    ierr = EPSSetWhichEigenpairs(eps,EPS_ALL);CHKERRQ(ierr);
    ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
    ierr = STSetType(st,STSINVERT);CHKERRQ(ierr);
    ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
//        ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    // save matrix size
    int mat_size;
    MatGetSize(A, &mat_size, NULL);
//        PetscPrintf(PETSC_COMM_WORLD," mat_size = %D\n", mat_size);

    // set requested number of eigenvalues here:
    // from documentation:
    // EPSSetDimensions(EPS eps, PetscInt nev, PetscInt ncv, PetscInt mpd);
    // The parameters ncv and mpd are intimately related, so that the user is advised to set one of them at most.
    // Normal usage is that (a) in cases where nev is small, the user sets ncv (a reasonable default is 2*nev);
    // and (b) in cases where nev is large, the user sets mpd.
    // The value of ncv should always be between nev and (nev+mpd), typically ncv=nev+mpd.
    // If nev is not too large, mpd=nev is a reasonable choice, otherwise a smaller value should be used.

    // ncv (number of column vectors): (i.e., the dimension of the subspace with which the eigensolver works).
    // This usually improves the convergence behavior at the expense of larger memory requirements.
    // mpd (maximum projected dimension): (read about the steps on page 45 of the manual)
    // The idea is to bound the size of the projected eigenproblem so that steps 3 and 4 work with a dimension
    // of mpd at most, while steps 1 and 2 still work with a bigger dimension, up to ncv.

    if(nev == 0) nev = mat_size; // if nev is not set, request all eigenvalues.
    if(ncv == 0) ncv = 2 * nev;  // if ncv is not set, set it to default.
    if(mpd == 0) mpd = nev;      // if mpd is not set, set it to default.
    EPSSetDimensions(eps, nev, ncv, mpd);

    // Set solver parameters at runtime

    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    //                    Solve the eigensystem
    //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    PetscReal threshold = 0.2 * mat_size;

    PetscReal inta = 0.1;
    PetscReal intb = 0.2;
    PetscReal interval_step = intb - inta;
//        PetscReal start_interval = inta;

    double *vec;
    int nlocal = 0;

//        store_eigenpairs = true; // set this to true to store eigenpairs in the wrapper class.
    if(store_eigenpairs) {
        // store eigenvalues and eigenvectors in the class.
        ierr = MatCreateVecs(A, nullptr, &xr);
        CHKERRQ(ierr);
        ierr = MatCreateVecs(A, nullptr, &xi);
        CHKERRQ(ierr);

        VecGetLocalSize(xr, &nlocal);

        auto temp_size = (unsigned long) floor(threshold);
        eig_val_real.resize(temp_size);
        eig_val_imag.resize(temp_size);
        eig_vec_real.resize(temp_size, std::vector<double>(nlocal));
        eig_vec_imag.resize(temp_size, std::vector<double>(nlocal));
    }

    unsigned int eig_num_prev = 0;

    while(eig_num < threshold) {
        // Set the interval and other settings for spectrum slicing
        ierr = EPSSetInterval(eps, inta, intb); CHKERRQ(ierr);

        ierr = EPSSolve(eps); CHKERRQ(ierr);
        ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

        ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
        eig_num_prev = eig_num;
        eig_num += nconv;
        ierr = PetscPrintf(PETSC_COMM_WORLD," Number of eigenvalues found in [%.2f, %.2f]: %D. Total: %D (%% %.2f)\n\n", inta, intb, nconv, eig_num, (float)100*eig_num/mat_size);CHKERRQ(ierr);

        // uncomment to store eigenpairs in the wrapper class
        if(store_eigenpairs) {
            if (eig_num > threshold) {
                eig_val_real.resize(eig_num);
                eig_val_imag.resize(eig_num);
                eig_vec_real.resize(eig_num, std::vector<double>(nlocal));
                eig_vec_imag.resize(eig_num, std::vector<double>(nlocal));
            }

            for (i = 0; i < nconv; i++) {
                EPSGetEigenpair(eps, i, &eig_val_real[i + eig_num_prev], &eig_val_imag[i + eig_num_prev], xr, xi);
                VecGetArray(xr, &vec);
                memcpy(&eig_vec_real[i + eig_num_prev][0], &vec[0], nlocal * sizeof(double));
                VecGetArray(xi, &vec);
                memcpy(&eig_vec_imag[i + eig_num_prev][0], &vec[0], nlocal * sizeof(double));
            }
        }

#ifdef __DEBUG1__
        // Optional: Get some information from the solver and display it
        if(verbose1){
            ierr = EPSGetIterationNumber(eps,&its);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);
            ierr = EPSGetST(eps,&st);CHKERRQ(ierr);
            ierr = STGetKSP(st,&ksp);CHKERRQ(ierr);
            ierr = KSPGetTotalIterations(ksp,&lits);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of linear iterations of the method: %D\n",lits);CHKERRQ(ierr);
            ierr = EPSGetType(eps,&type);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type);CHKERRQ(ierr);
            ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);
            ierr = EPSGetTolerances(eps,&tol,&maxit);CHKERRQ(ierr);
            ierr = PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
            ierr = EPSReasonView(eps,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }

        // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //                  Display solution and clean up
        //   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        // Show detailed info if verbose is true.

        if(verbose2){
            ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
            ierr = EPSErrorView(eps,EPS_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }
#endif

        inta += interval_step;
        intb += interval_step;
    }

    ierr = PetscPrintf(PETSC_COMM_WORLD," -------------------------------------------------------\n");CHKERRQ(ierr);
//        ierr = PetscPrintf(PETSC_COMM_WORLD," Total number of eigenvalues found in [%.2f, %.2f]: %D (%% %.2f)\n", start_interval, inta, eig_num, (float)100*eig_num/mat_size);CHKERRQ(ierr);

    // Free work space

    ierr = EPSDestroy(&eps);CHKERRQ(ierr);
    if(store_eigenpairs){
        ierr = VecDestroy(&xr);CHKERRQ(ierr);
        ierr = VecDestroy(&xi);CHKERRQ(ierr);
    }
    if (nini > 0) {
        ierr = VecDestroyVecs(nini,&Iv);CHKERRQ(ierr);
    }
    if (ncon > 0) {
        ierr = VecDestroyVecs(ncon,&Cv);CHKERRQ(ierr);
    }

    return 0;
}


void eigen_mm::print_eig_val_real(){
    print_vector(eig_val_real, 0, "eigenvalues (real part):", comm);
}

void eigen_mm::print_eig_val_imag(){
    print_vector(eig_val_imag, 0, "eigenvalues (imaginary part):", comm);
}

void eigen_mm::print_eig_val(){

    int rank;
    MPI_Comm_rank(comm, &rank);

    if(!store_eigenpairs){
        if(rank==0) printf("set store_eigenpairs to true to store eigenpairs in the wrapper class.\n");
    }

    MPI_Barrier(comm);
    if(rank==0){
        printf("\neigenvalues, size = %ld: \n", eig_val_real.size());
        for (unsigned i = 0; i < eig_val_real.size(); i++) {
            std::cout << i << "\t" << eig_val_real[i] << " + " << eig_val_imag[i] << "i" << std::endl;
        }
        printf("\n");
    }

    MPI_Barrier(comm);
}

void eigen_mm::print_eig_vec_real(int ran){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)
    std::string text = "real part of eigenvector ";
    for(int i = 0; i < eig_vec_real.size(); i++){
        print_vector(eig_vec_real[i], ran, text + std::to_string(i), comm);
    }
}

void eigen_mm::print_eig_vec_imag(int ran){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)
    std::string text = "imaginary part of eigenvector ";
    for(int i = 0; i < eig_vec_imag.size(); i++){
        print_vector(eig_vec_imag[i], ran, text + std::to_string(i), comm);
    }
}

void eigen_mm::print_eig_vec(int ran){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!store_eigenpairs){
        if(rank==0) printf("set store_eigenpairs to true to store eigenpairs in the wrapper class.\n");
    }

    MPI_Barrier(comm);
    for(int i = 0; i < eig_vec_real.size(); i++) {

        if (ran >= 0) {
            if (rank == ran) {
                printf("\neigenvector %d on proc = %d, size = %ld: \n", i, rank, eig_vec_real.size());
                for (unsigned j = 0; j < eig_vec_real[i].size(); j++) {
                    std::cout << j << "\t" << eig_vec_real[i][j] << " + " << eig_vec_imag[i][j] << "i" << std::endl;
                }
                printf("\n");
            }
        } else {
            for (unsigned proc = 0; proc < nprocs; proc++) {
                MPI_Barrier(comm);
                if (rank == proc) {
                    printf("\neigenvector %d on proc = %d, size = %ld: \n", i, proc, eig_vec_real.size());
                    for (unsigned j = 0; j < eig_vec_real[i].size(); j++) {
                        std::cout << j << "\t" << eig_vec_real[i][j] << " + " << eig_vec_imag[i][j] << "i" << std::endl;
                    }
                    printf("\n");
                }
                MPI_Barrier(comm);
            }
        }

    }
    MPI_Barrier(comm);
}


int eigen_mm::viewA(){

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);

    return 0;
}

int eigen_mm::viewB(){

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(B, viewer);
    PetscViewerDestroy(&viewer);

    return 0;
}


int eigen_mm::get_eig_num(){
    return eig_num;
}

double* eigen_mm::get_eig_val_real(){
    return &eig_val_real[0];
}

double* eigen_mm::get_eig_val_imag(){
    return &eig_val_imag[0];
}

double* eigen_mm::get_eig_vec_real(int i){
    return &eig_vec_real[i][0];
}

double* eigen_mm::get_eig_vec_imag(int i){
    return &eig_vec_imag[i][0];
}