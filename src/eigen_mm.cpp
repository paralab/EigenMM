#include "eigen_mm.h"

// ===============================================
// Source: https://en.wikipedia.org/wiki/Adler-32
const int MOD_ADLER = 65521;
uint32_t adler32(unsigned char *data, size_t len)
{
    uint32_t a = 1, b = 0;
    size_t index;

    for (index = 0; index < len; ++index)
    {
        a = (a + data[index]) % MOD_ADLER;
        b = (b + a) % MOD_ADLER;
    }
    
    return (b << 16) | a;
}
// ===============================================

eigen_mm::eigen_mm() = default;

eigen_mm::~eigen_mm(){
    //MatDestroy(&K);
    //MatDestroy(&M);
};

int eigen_mm::init(Mat &K_in, Mat &M_in, SolverOptions *opts_in)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    opts = *opts_in;

    // Initialize World Communicator
    MPI_Comm_rank(PETSC_COMM_WORLD, &(node.worldrank));
    MPI_Comm_size(PETSC_COMM_WORLD, &(node.worldsize));

    // Initialize Evaluator Communicator
    int resultlen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &resultlen);
    uint32_t processor_id = adler32((unsigned char *) &processor_name[0], resultlen*sizeof(char));
    MPI_Comm_split(PETSC_COMM_WORLD, processor_id, node.worldrank, &(node.comm));
    MPI_Comm_size(node.comm, &(node.size));
    MPI_Comm_rank(node.comm, &(node.rank));
    node.id = node.worldrank / node.size;
    node.viewer = PETSC_VIEWER_STDOUT_(node.comm);

    // Initialize Row Communicator
    node.rowid = node.worldrank % node.size;
    MPI_Comm_split(PETSC_COMM_WORLD, node.rowid, node.worldrank, &(node.rowcomm));
    MPI_Comm_rank(node.rowcomm, &(node.rowrank));
    MPI_Comm_size(node.rowcomm, &(node.rowsize)); 

    node.nevaluators = node.worldsize / node.size;
    opts.set_nevaluators(node.nevaluators);
    opts.set_totalsubproblems(opts.nevaluators() * opts.subproblemsperevaluator());

    scatterInputMats(K_in, M_in);

    node.neval = 0;

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report init time
    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(init) Elapsed: %lf, Start: %lf, End: %lf\n", elapsed, start_time, end_time);
}

int eigen_mm::solve(Mat *V_out, Vec *lambda_out)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "finding upper bound\n");
    findUpperBound();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming subproblems\n");
    formSubproblems();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "solving subproblems\n");
    PetscInt neval = solveSubproblems();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming eigenbasis\n");
    formEigenbasis(neval);
    (*V_out) = V;
    (*lambda_out) = lambda;

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(solve) Elapsed: %lf, Start: %lf, End: %lf\n", elapsed, start_time, end_time);
}

Mat& eigen_mm::getK(){
    return K;
}

Mat& eigen_mm::getM(){
    return M;
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


int eigen_mm::viewK(){

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(K, viewer);
    PetscViewerDestroy(&viewer);

    return 0;
}

int eigen_mm::viewM(){

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(M, viewer);
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


// ==================== World ====================
void eigen_mm::findUpperBound()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    PetscInt oldp = opts.p();
    PetscInt oldnv = opts.nv();
    if (oldp > 0)
    {
        opts.set_p(100);
        opts.set_nv(30);
    }

    PetscReal lR;
    PetscInt rev, in_rev, iter;

    iter = 0;
    lR = (1 << node.id) * opts.R();
    rev = (opts.p() > 0) ? computeDev_approximate(lR, 2*lR, PETSC_TRUE)
                         : computeDev_exact(lR, PETSC_TRUE);

    for (int i = 0; i < node.nevaluators; i++)
    {
        in_rev = rev;
        MPI_Bcast(&in_rev, 1, MPIU_INT, i*node.size, PETSC_COMM_WORLD);
        if (in_rev <= 0)
        {
            lR = (1 << i) * opts.R();
            break;
        }
    }
    while(in_rev > 0)
    {
        iter++;
        lR = (1 << (node.id + iter*node.nevaluators)) * opts.R();
        rev = (opts.p() > 0) ? computeDev_approximate(lR, 2*lR, PETSC_TRUE)
                             : computeDev_exact(lR, PETSC_TRUE);
        for (int i = 0; i < node.nevaluators; i++)
        {
            in_rev = rev;
            MPI_Bcast(&in_rev, 1, MPIU_REAL, i*node.size, PETSC_COMM_WORLD);
            if(in_rev <= 0) 
            {
                lR = (1 << (i + iter*node.nevaluators)) * opts.R();
                break;
            }
        }
    }

    opts.set_R(lR);
    opts.set_p(oldp);
    opts.set_nv(oldnv);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report findUpperBound timing
    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf\n", elapsed, start_time, end_time, (double) lR);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(findUpperBound) Elapsed: %lf, Start: %lf, End: %lf, Upper Bound: %lf\n", elapsed, start_time, end_time, (double) lR);
}
void eigen_mm::formSubproblems()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    PetscReal leafintervals[10000];
    PetscReal splitintervals[10000];
    PetscInt  leafcounts[10000];

    PetscInt fleft, fright, noutput, split_index, leaf_index;
    PetscReal split;
    
    splitintervals[0] = opts.L();
    splitintervals[1] = opts.R();

    PetscInt nleaves = 0;
    int current_split = 1;
    int split_max_depth = 20;
    int job0 = node.id;
    int jobstride = node.nevaluators;
    int njobs = 1;

    while (nleaves       < opts.totalsubproblems() && 
           njobs         > 0 && 
           current_split < split_max_depth)
    {
        for (int job = job0; job < njobs; job += jobstride)
        {
            splitSubProblem(splitintervals[2*job + 0], splitintervals[2*job + 1], 
                &split, &fleft, &fright);
            leafintervals[2*nleaves + 4*job + 0] = splitintervals[2*job + 0];
            leafintervals[2*nleaves + 4*job + 1] = split;
            leafintervals[2*nleaves + 4*job + 2] = split;
            leafintervals[2*nleaves + 4*job + 3] = splitintervals[2*job + 1];
            leafcounts[nleaves + 2*job + 0] = fleft;
            leafcounts[nleaves + 2*job + 1] = fright;
        }

        for (int job = 0; job < njobs; job++)
        {
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 0], 1, MPIU_REAL, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 1], 1, MPIU_REAL, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 2], 1, MPIU_REAL, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 3], 1, MPIU_REAL, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
            MPI_Bcast(&leafcounts[nleaves + 2*job + 0],    1, MPIU_INT, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
            MPI_Bcast(&leafcounts[nleaves + 2*job + 1],    1, MPIU_INT, (job % node.nevaluators)*node.size, PETSC_COMM_WORLD);
        }
        MPI_Barrier(PETSC_COMM_WORLD);

        noutput     = 2*njobs;
        split_index = 0;
        leaf_index  = 0;
        for (int i = 0; i < noutput; i++)
        {
            if (leafcounts[nleaves + i] > opts.nevt() &&
                nleaves + noutput < opts.totalsubproblems() &&
                current_split < split_max_depth - 1)
            {
                splitintervals[2*split_index + 0] = leafintervals[2*nleaves + 2*i + 0];
                splitintervals[2*split_index + 1] = leafintervals[2*nleaves + 2*i + 1];
                split_index++;
            }
            else
            {
                leafintervals[2*(nleaves + noutput + leaf_index) + 0] = leafintervals[2*(nleaves + i) + 0];
                leafintervals[2*(nleaves + noutput + leaf_index) + 1] = leafintervals[2*(nleaves + i) + 1];
                leafcounts[nleaves + noutput + leaf_index] = leafcounts[nleaves + i];
                leaf_index++;
            }
        }
        for (int i = 0; i < leaf_index; i++)
        {
            leafintervals[2*(nleaves + i) + 0] = leafintervals[2*(nleaves + noutput + i) + 0];
            leafintervals[2*(nleaves + i) + 1] = leafintervals[2*(nleaves + noutput + i) + 1];
            leafcounts[nleaves + i] = leafcounts[nleaves + noutput + i];
        }
        nleaves += leaf_index;
        njobs   = split_index;
        MPI_Barrier(PETSC_COMM_WORLD);

        current_split++;
    }
    for (int i = 0; i < nleaves; i++)
    {
        intervals.push_back(leafintervals[2*i+0]);
        intervals.push_back(leafintervals[2*i+1]);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report formSubproblems timing
    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %d\n", elapsed, start_time, end_time, (int) nleaves);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(formSubproblems) Elapsed: %lf, Start: %lf, End: %lf, Total Subproblems: %d\n", elapsed, start_time, end_time, (int) nleaves);
}
PetscInt eigen_mm::solveSubproblems()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    int job0       = node.id;
    int jobstride  = node.nevaluators;
    int njobs      = intervals.size() / 2;
    PetscInt neval = 0;
    PetscInt in_neval;
    for (int job = job0; job < njobs; job += jobstride)
    {
        neval += solveSubProblem(intervals[2*job+0], intervals[2*job+1], job);
    }
    for (int i = 1; i < node.nevaluators; i++)
    {
        in_neval = neval;
        MPI_Bcast(&in_neval, 1, MPIU_INT, i*node.size, PETSC_COMM_WORLD);
        if (node.id == 0) neval += in_neval;
    }
    MPI_Bcast(&neval, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %d\n", elapsed, start_time, end_time, (int) neval);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(solveSubproblems) Elapsed: %lf, Start: %lf, End: %lf, Total Eigenpairs Found: %d\n", elapsed, start_time, end_time, (int) neval);

    return neval;
}
void eigen_mm::formEigenbasis(PetscInt neval)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Process p has block b:
    //  size of block b: (lm by ln)
    //  ln is the number of eigenpairs computed by process p's evaluator
    //  lm is lv_data.size() / ln
    PetscInt N;
    MatGetSize(K, &N, NULL);

    PetscInt ln = node.neval;
    PetscInt lm = lv_data.size() / ln;

    // transpose lv_data?
    std::vector<PetscScalar> lv_data_transpose(lv_data.size());
    for (int row = 0; row < lm; row++)
        for (int col = 0; col < ln; col++)
            lv_data_transpose[col + row*ln] = lv_data[row + col*lm];

    // Determine row0 for each process
    //   reduce over lm on evaluator communicator
    PetscInt all_lm[node.size];
    PetscInt all_row0[node.size];
    all_lm[node.rank] = lm;
    for (int i = 0; i < node.size; i++)
        MPI_Bcast(&all_lm[i], 1, MPIU_INT, i, node.comm);
    all_row0[0] = 0;
    for (int i = 1; i < node.size; i++)
        all_row0[i] = all_row0[i-1] + all_lm[i-1];
    PetscInt row0 = all_row0[node.rank];

    // Determine col0 for each process
    //   reduce over ln on row communicator
    PetscInt all_ln[node.rowsize];
    PetscInt all_col0[node.rowsize];
    all_ln[node.rowrank] = ln;
    for (int i = 0; i < node.rowsize; i++)
        MPI_Bcast(&all_ln[i], 1, MPIU_INT, i, node.rowcomm);
    all_col0[0] = 0;
    for (int i = 1; i < node.rowsize; i++)
        all_col0[i] = all_col0[i-1] + all_ln[i-1];
    PetscInt col0 = all_col0[node.rowrank];

    // allocate idxm and idxn
    PetscInt idxm[lm];
    PetscInt idxn[ln];
    int idx = 0;
    for (int row = 0; row < lm; row++)
        idxm[row] = row0 + row;
    for (int col = 0; col < ln; col++)
        idxn[col] = col0 + col;

    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, 
        PETSC_DECIDE, N, neval, NULL, &V);
    MatSetUp(V);
    MatSetValues(V, lm, idxm, ln, idxn, &lv_data_transpose[0], INSERT_VALUES);
    MPI_Barrier(PETSC_COMM_WORLD);
    MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);

    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, neval, &lambda);
    VecSetValues(lambda, ln, idxn, &lambda_data[0], INSERT_VALUES);
    MPI_Barrier(PETSC_COMM_WORLD);
    VecAssemblyBegin(lambda);
    VecAssemblyEnd(lambda);

    if(opts.savelambda())
    {
        PetscViewer viewer;
        char filenamelambda[1024];
        sprintf(filenamelambda, "%slambda%d", opts.lambda_filepath(), (int)N);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenamelambda, 
            FILE_MODE_WRITE, &viewer);
        VecView(lambda, viewer);
        PetscViewerDestroy(&viewer);
    }

    if(opts.saveV())
    {
        PetscViewer viewer;
        char filenameV[1024];
        sprintf(filenameV, "%sV%d", opts.V_filepath(), (int)N);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenameV, 
            FILE_MODE_WRITE, &viewer);
        PetscViewerPushFormat(viewer, PETSC_VIEWER_NATIVE);
        MatView(V, viewer);
        PetscViewerDestroy(&viewer);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(formEigenbasis) Elapsed: %lf, Start: %lf, End: %lf\n", elapsed, start_time, end_time);
}
// ================================================

// ==================== Evaluator ====================
void eigen_mm::scatterInputMats(Mat &K_in, Mat &M_in)
{
    MatCreateRedundantMatrix(K_in, node.nevaluators, node.comm, MAT_INITIAL_MATRIX, &K);
    MatCreateRedundantMatrix(M_in, node.nevaluators, node.comm, MAT_INITIAL_MATRIX, &M);
}
PetscInt eigen_mm::solveSubProblem(PetscReal a, PetscReal b, int job)
{
    MPI_Barrier(node.comm);
    double start_time = MPI_Wtime();

    if(opts.debug()) PetscPrintf(node.comm, "Solving interval [%lf, %lf]\n", a, b);

    // Set up solver
    PetscInt nconv;
    EPS eps;
    char subinterval_string[1024] = "";
    sprintf(subinterval_string, "-eps_interval %lf,%lf", (double)a, (double)b);
    PetscOptionsInsertString(nullptr, subinterval_string);
    PetscOptionsInsertString(nullptr, "-st_type sinvert");
    PetscOptionsInsertString(nullptr, "-st_ksp_type preonly");
    PetscOptionsInsertString(nullptr, "-st_pc_type cholesky");
    PetscOptionsInsertString(nullptr, "-st_pc_factor_mat_solver_package mumps");
    PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_13 1");
    PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_14 80");
    EPSCreate(node.comm,&eps);
    EPSSetOperators(eps,K,M);
    EPSSetProblemType(eps,EPS_GHEP);
    EPSSetFromOptions(eps);

    // Solve
    EPSSolve(eps);

    // Process results
    EPSGetConverged(eps, &nconv); 
    
    Vec v;
    PetscInt N, size;
    PetscReal lam, *v_data;

    MatGetSize(K, &N, NULL);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &v);
    VecGetLocalSize(v, &size);
    for (int i = 0; i < nconv; i++)
    {
        EPSGetEigenpair(eps, i, &lam, NULL, v, NULL);
        lambda_data.push_back(lam);
        VecGetArray(v, &v_data);
        for (int j = 0; j < size; j++) lv_data.push_back(v_data[j]);
        VecRestoreArray(v, &v_data);
    }
 
    // Clean up solver
    VecDestroy(&v);
    EPSDestroy(&eps);

    node.neval += nconv;

    MPI_Barrier(node.comm);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(node.comm, "%d %d %lf %lf %lf %d\n", node.id, job, elapsed, start_time, end_time, (int) nconv);
    else
        PetscPrintf(node.comm, "(solveSubproblem) Evaluator: %d, Subinterval: %d, Elapsed: %lf, Start: %lf, End: %lf, Number of Eigenpairs: %d\n", node.id, job, elapsed, start_time, end_time, (int) nconv);

    return nconv;
}
void eigen_mm::splitSubProblem(PetscReal a, PetscReal b, 
        PetscReal *c, PetscInt *out_ec_left, PetscInt *out_ec_right)
{
    MPI_Barrier(node.comm);
    double start_time = MPI_Wtime();

    PetscReal bleft, bright, split, ratio;
    PetscInt iter, ec_left, ec_right;

    bleft  = a;
    bright = b;
    iter   = 1;
    ratio  = 0.0;
    while (ratio < opts.splittol() && iter < opts.splitmaxiters())
    {
        split = (bleft + bright) / 2.0;
        countInterval(a, split, &ec_left);
        countInterval(split, b, &ec_right);
        if (ec_left < ec_right) { bleft  = split; }
        else                    { bright = split; }
        ratio = (std::max(ec_left, ec_right) > 0) ? fabs(std::min(ec_left, ec_right) / std::max(ec_left, ec_right)) : 1.0;
        iter++;
    }
    c[0]            = split;
    out_ec_left[0]  = ec_left;
    out_ec_right[0] = ec_right;

    MPI_Barrier(node.comm);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(node.comm, "%d %lf %lf %lf %d %d %lf %lf %lf", node.id, elapsed, start_time, end_time, (int) ec_left, (int) ec_right, (double) a, (double) b, (double) split);
    else
        PetscPrintf(node.comm, "(splitSubproblem) Evaluator: %d, Elapsed: %lf, Start: %lf, End: %lf, Left Count: %d, Right Count: %d, a = %lf, b = %lf, c = %lf\n", node.id, elapsed, start_time, end_time, (int) ec_left, (int) ec_right, (double) a, (double) b, (double) split);
}
PetscInt eigen_mm::computeDev_exact(PetscReal a, PetscBool rl)
{
    PetscInt N, retval;

    MatGetSize(K, &N, NULL);

    Mat A, L;
    MatConvert(M, MATSAME, MAT_INITIAL_MATRIX, &A);
    MatAYPX(A, -a, K, DIFFERENT_NONZERO_PATTERN);

    MatGetFactor(A, "mumps", MAT_FACTOR_CHOLESKY, &L);
    MatMumpsSetIcntl(L, 13, 1);
    MatCholeskyFactorSymbolic(L, A, 0, 0);
    MatCholeskyFactorNumeric(L, A, 0);
    MatMumpsGetInfog(L, 12, &retval);

    MatDestroy(&L);
    MatDestroy(&A);

    MPI_Barrier(node.comm);

    return (rl) ? N - retval : retval;
}
PetscInt eigen_mm::computeDev_approximate(PetscReal a, PetscReal U, PetscBool rl)
{
    Mat A;
    Vec vk, vk2, wm2, wm1, w;

    PetscRandom r;
    PetscReal vknorm, vkdot, radius, gamma, rev;
    PetscInt N, retval;

    PetscRandomCreate(node.comm, &r);
    PetscRandomSetType(r, PETSCRAND);
    PetscRandomSetInterval(r, 0, 1);

    MatConvert(M, MATSAME, MAT_INITIAL_MATRIX, &A);
    MatAYPX(A, -a, K, DIFFERENT_NONZERO_PATTERN);
    MatGetSize(A, &N, NULL);

    VecCreateMPI(node.comm, PETSC_DECIDE, N, &vk);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &vk2);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &wm2);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &wm1);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &w);

    radius = (PetscReal) 2.0 * computeRadius(A);
    MatScale(A, (PetscScalar) 1.0/radius);

    rev = 0;
    for (int k = 1; k <= opts.nv(); k++)
    {
        PetscInt n;
        PetscScalar *x1, *x2;

        // vk = normalize(randn(N,1))
        VecSetRandom(vk, r);
        VecSetRandom(vk2, r);
        VecGetLocalSize(vk, &n);
        VecGetArray(vk, &x1);
        VecGetArray(vk2, &x2);
        for (int i = 0; i < n; i++) { x1[i] = sqrt(-2*log(x1[i]))*cos(2*PI*x2[i]); x2[i] = 0.0; }
        VecRestoreArray(vk, &x1);
        VecRestoreArray(vk2, &x2);

        for (int j = 0; j <= opts.p(); j++)
        {
            if (j == 0)
            {
                gamma = 0.5;
                VecCopy(vk, w);
                VecCopy(vk, wm2);
            }
            else if (j == 1)
            {
                gamma = 2.0/PI;
                gamma *= (a > 0.5*U) ? -1 : 1;
                MatMult(A,vk,w);
                VecCopy(w, wm1);
            }
            else
            {
                gamma = 2.0/PI * sin(j*PI/2.0)/j;
                gamma *= (a > 0.5*U) ? -1 : 1;
                MatMult(A,wm1,w);
                VecScale(w, 2);
                VecAXPY(w, -1, wm2);
                VecCopy(wm1, wm2);
                VecCopy(w, wm1);
            }
            VecAXPY(vk2, gamma, w);
        }
        VecDot(vk, vk2, &vkdot);
        rev += vkdot;
    }
    retval = (PetscInt) round((PetscReal)1/(PetscReal)opts.nv() * rev);
    retval = (a > U/2)  ? N - retval : retval;
    retval = ((bool)rl) ? retval : N - retval;

    MatDestroy(&A);
    VecDestroy(&vk);
    VecDestroy(&vk2);
    VecDestroy(&wm2);
    VecDestroy(&wm1);
    VecDestroy(&w);

    MPI_Barrier(node.comm);

    return retval;
}
PetscReal eigen_mm::computeRadius(Mat &A)
{
    Vec x, Sx;
    PetscRandom r;
    PetscReal e, e0, normx, normsx;
    PetscInt N, iter;

    PetscRandomCreate(node.comm, &r);
    PetscRandomSetType(r, PETSCRAND);
    PetscRandomSetInterval(r, -1, 1);

    MatGetSize(A, &N, NULL);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &x);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &Sx);

    VecSetRandom(x, r);
    VecCopy(x, Sx);
    VecAbs(Sx);
    VecPointwiseDivide(Sx, x, Sx);
    MatMult(A, Sx, x);
    VecAbs(x);
    VecNorm(x, NORM_2, &e);
    VecScale(x, 1/e);
    e0 = 0;
    iter = 1;
    while (abs(e - e0) > opts.radtol()*e && iter <= opts.raditers())
    {
        e0 = e;
        MatMult(A, x, Sx);
        MatMult(A, Sx, x);
        VecNorm(x, NORM_2, &normx);
        VecNorm(Sx, NORM_2, &normsx);
        e = normx / normsx;
        VecScale(x, 1/normx);
        
        iter++;
    }

    VecDestroy(&x);
    VecDestroy(&Sx);

    return e;
}
void eigen_mm::countInterval(PetscReal a, PetscReal b, 
    PetscInt *count)
{
    PetscInt reva = (opts.p() > 0) ? computeDev_approximate(a, opts.R(), PETSC_TRUE)
                                   : computeDev_exact(a, PETSC_TRUE);
    PetscInt revb = (opts.p() > 0) ? computeDev_approximate(b, opts.R(), PETSC_TRUE)
                                   : computeDev_exact(b, PETSC_TRUE);
    count[0] = reva - revb;
    MPI_Barrier(node.comm);
}
// ===================================================