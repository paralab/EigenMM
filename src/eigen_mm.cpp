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

void eigen_mm::checkCorrectness()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);
    residuals.resize(neval);

    Mat residual, temp;
    MatConvert(V, MATSAME, MAT_INITIAL_MATRIX, &temp);
    MatDiagonalScale(temp, NULL, lambda);
    MatMatMult(M_global, temp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &residual);
    MatMatMult(K_global, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &temp);
    MatAYPX(residual, -1, temp, DIFFERENT_NONZERO_PATTERN);

    MatGetColumnNorms(residual, NORM_2, &residuals[0]);

    PetscReal minnorm = MPIU_MAX;
    PetscReal maxnorm = 0.0;
    PetscReal avgnorm = 0.0;
    for (int k = 0; k < neval; k++)
    {
        minnorm = (minnorm < residuals[k]) ? minnorm : residuals[k];
        maxnorm = (maxnorm > residuals[k]) ? maxnorm : residuals[k];
        avgnorm += residuals[k];
    }
    avgnorm /= neval;

    MatDestroy(&residual);
    MatDestroy(&temp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    std::ofstream output_file(opts.correctness_filename());
    std::ostream_iterator<PetscReal> output_iterator(output_file, "\n");
    std::copy(residuals.begin(), residuals.end(), output_iterator);

    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf %lf %lf\n", elapsed, start_time, end_time, (double) minnorm, (double) maxnorm, (double) avgnorm);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(checkCorrectness) ||A*vk - lambdak*M*vk|| (min/max/avg) = (%.16lf / %.16lf / %.16lf), Elapsed = %lf, Start = %lf, End = %lf\n", (double) minnorm, (double) maxnorm, (double) avgnorm, elapsed, start_time, end_time);
}

void eigen_mm::checkOrthogonality()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // residual = eye(neval) - V' * M * V

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    Vec ones;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, neval, &ones);
    VecSet(ones, -1.0);

    Mat residual, temp;
    MatMatMult(M, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp);
    MatTransposeMatMult(V, temp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &residual);
    MatDiagonalSet(residual, ones, ADD_VALUES);

    PetscScalar norm;
    MatNorm(residual, NORM_FROBENIUS, &norm);

    VecDestroy(&ones);
    MatDestroy(&residual);
    MatDestroy(&temp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf", elapsed, start_time, end_time, (double) norm);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(Orthogonality Check) ||I - V'*V|| = %lf, Elapsed = %lf, Start = %lf, End = %lf\n", (double) norm, elapsed, start_time, end_time);
}

eigen_mm::eigen_mm() = default;

eigen_mm::~eigen_mm(){
    //MatDestroy(&K);
    //MatDestroy(&M);
};

int eigen_mm::init(Mat &K_in, Mat &M_in, SolverOptions *opts_in)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    K_global = K_in;
    M_global = M_in;

    opts = *opts_in;

    // Initialize World Communicator
    MPI_Comm_rank(PETSC_COMM_WORLD, &(node.worldrank));
    MPI_Comm_size(PETSC_COMM_WORLD, &(node.worldsize));

    // Initialize Evaluator Communicator
    int resultlen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &resultlen);
    uint32_t processor_id = adler32((unsigned char *) &processor_name[0], resultlen*sizeof(char));
    uint32_t evaluator_id;

    // Gather processor ids to root
    int count = 0;
    std::vector<uint32_t> processor_ids;
    std::vector<uint32_t> evaluator_ids;
    std::map<uint32_t, uint32_t> processor_to_evaluator;
    if (node.worldrank == 0) processor_ids.resize(node.worldsize);
    if (node.worldrank == 0) evaluator_ids.resize(node.worldsize);
    MPI_Gather(&processor_id, 1, MPI_UINT32_T, &processor_ids[0], 1, MPI_UINT32_T, 0, PETSC_COMM_WORLD);
    if (node.worldrank == 0)
    {
        for (int i = 0; i < node.worldsize; i++)
        {
            uint32_t key = processor_ids[i];
            if (processor_to_evaluator.count(key) == 0)
            {
                processor_to_evaluator[key] = count / opts.nodesperevaluator();
                count++;
            }
        }
        for (int i = 0; i < node.worldsize; i++)
        {
            evaluator_ids[i] = processor_to_evaluator[processor_ids[i]];
        }
    }
    MPI_Scatter(&evaluator_ids[0], 1, MPI_UINT32_T, &evaluator_id, 1, MPI_UINT32_T, 0, PETSC_COMM_WORLD);
    if (node.worldrank == 0) count = processor_to_evaluator.size();
    MPI_Bcast(&count, 1, MPI_INT, 0, PETSC_COMM_WORLD);

    MPI_Comm_split(PETSC_COMM_WORLD, evaluator_id, node.worldrank, &(node.comm));
    MPI_Comm_size(node.comm, &(node.size));
    MPI_Comm_rank(node.comm, &(node.rank));
    node.id = node.worldrank / node.size;
    node.viewer = PETSC_VIEWER_STDOUT_(node.comm);

    // Initialize Row Communicator
    node.rowid = node.worldrank % node.size;
    MPI_Comm_split(PETSC_COMM_WORLD, node.rowid, node.worldrank, &(node.rowcomm));
    MPI_Comm_rank(node.rowcomm, &(node.rowrank));
    MPI_Comm_size(node.rowcomm, &(node.rowsize));

    node.nevaluators = count / opts.nodesperevaluator();
    opts.set_nevaluators(node.nevaluators);
    opts.set_totalsubproblems(opts.nevaluators() * opts.subproblemsperevaluator());

    if(opts.save_operators())
    {
        char K_filename[1024];
        char M_filename[1024];
        sprintf(K_filename, "%s_K", opts.operators_filename());
        sprintf(M_filename, "%s_M", opts.operators_filename());

        PetscViewer viewer;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, K_filename, 
            FILE_MODE_WRITE, &viewer);
        MatView(K_in, viewer);
        PetscViewerDestroy(&viewer);

        PetscViewerBinaryOpen(PETSC_COMM_WORLD, M_filename, 
            FILE_MODE_WRITE, &viewer);
        MatView(M_in, viewer);
        PetscViewerDestroy(&viewer);
    }

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

    PetscInt N;
    MatGetSize(K_global, &N, NULL);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, NULL, &V);
    MatSetUp(V);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &lambda);

    if (opts.R() <= opts.L() && opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "finding upper bound\n");
    if (opts.R() <= opts.L()) findUpperBound();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming subproblems\n");
    formSubproblems();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "solving subproblems\n");
    PetscInt neval = solveSubproblems();
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming eigenbasis\n");
    formEigenbasis(neval);
    if(opts.save_correctness()) checkCorrectness();
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

PetscReal eigen_mm::find_amax(PetscInt k)
{
    PetscReal bleft, bright, split;
    PetscInt iter, ec, err;

    bleft = 0;
    bright = opts.R();
    iter = 1;
    while(abs(err) > opts.nevt() && iter < opts.splitmaxiters())
    {
        split = (bleft + bright) / 2;
        countInterval(split, opts.R(), &ec);
        err = ec - k;
        if (err < 0) { bleft  = split; }
        else         { bright = split; }
        iter++;
    }
    return split;
}
PetscReal eigen_mm::find_b(PetscInt k, PetscReal a)
{
    PetscReal bleft, bright, split;
    PetscInt iter, ec, err;

    bleft = a;
    bright = opts.R();
    iter = 1;
    while(abs(err) > opts.nevt() && iter < opts.splitmaxiters())
    {
        split = (bleft + bright) / 2;
        countInterval(a, split, &ec);
        err = ec - k;
        if (err < 0) { bleft  = split; }
        else         { bright = split; }
        iter++;
    }
    return split;
}

int eigen_mm::solvetime_exp()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // for i = 0 : nk - 1
    //   compute amax using k[i] and R
    //   for j = 0 : ns - 1
    //     find b[j] such that the interval [a[j], b[j]] contains k[i] eigenpairs
    //     solve the interval [a[j], b[j]] and store elapsed in e[i*ns + j]

    // obtain (L, R, ns) from solver options (use nv for ns since p will be 0)

    // hard code kmax, kmin, nk
    PetscInt kmin = 100;
    PetscInt kmax = 4096;
    PetscInt nk = 10;
    PetscInt kstep = (kmax - kmin) / (nk - 1);

    // initialize arrays a[ns], b[ns], k[nk], e[nk*ns]
    std::vector<PetscReal> a(opts.nv() * nk);
    std::vector<PetscReal> b(opts.nv() * nk);
    std::vector<PetscReal> e(opts.nv() * nk);
    std::vector<PetscInt>  d(opts.nv() * nk);
    std::vector<PetscInt>  k(nk);

    PetscReal amin = opts.L();
    PetscReal amax, astep;

    PetscPrintf(PETSC_COMM_WORLD, "Starting experiment:\n");
    for (int i = 0; i < nk; i++)
    {
        amax = find_amax(k[i]);
        astep = (amax - amin) / (opts.nv() - 1);
        if (opts.terse())
            PetscPrintf(PETSC_COMM_WORLD, "%d %lf %lf\n", k[i], (double)opts.L(), (double)amax);
        else
            PetscPrintf(PETSC_COMM_WORLD, "  (k = %d) Checking starting points in interval [%lf, %lf]\n", k[i], (double)opts.L(), (double)amax);
        for (int j = 0; j < opts.nv(); j++)
        {
            int idx = i*opts.nv() + j;
            a[idx] = amin + astep * j;
            b[idx] = find_b(k[i], a[idx]);

            MPI_Barrier(PETSC_COMM_WORLD);
            double exp_start_time = MPI_Wtime();

            PetscReal interval[2] = {a[idx], b[idx]};
            PetscInt neval = solveSubProblem(interval, 0);

            MPI_Barrier(PETSC_COMM_WORLD);
            double exp_end_time = MPI_Wtime();
            d[idx] = neval - k[i];
            e[idx] = exp_end_time - exp_start_time;

            if (opts.terse())
                PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %d %d\n", 
                    (double)a[idx], (double)b[idx], (double)e[idx], neval, d[idx]);
            else
                PetscPrintf(PETSC_COMM_WORLD, "    Interval [%lf, %lf]: Time elapsed: %lf, Eigenvalues found: %d, Difference from expected: %d\n", 
                    (double)a[idx], (double)b[idx], (double)e[idx], neval, d[idx]);
        }
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    PetscPrintf(PETSC_COMM_WORLD, "Finished with all experiments in %lf seconds.\n", elapsed);
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

    if (node.id == 0) MatView(K, node.viewer);

    return 0;
}

int eigen_mm::viewM(){

    if (node.id == 0) MatView(M, node.viewer);
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

    PetscReal lR;

    if (node.id == 0)
    {
        Mat F;
        KSP ksp;
        PC pc;
        Vec x, Sx, b;
        PetscRandom r;
        PetscReal e, e0, normx, normsx;
        PetscInt N, iter;

        KSPCreate(node.comm, &ksp);
        KSPSetOperators(ksp, M, M);
        KSPSetType(ksp, KSPPREONLY);
        KSPGetPC(ksp, &pc);
        PCSetType(pc, PCCHOLESKY);
        PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS);
        PCFactorSetUpMatSolverPackage(pc);
        PCFactorGetMatrix(pc, &F);
        MatMumpsSetIcntl(F, 13, 1);
        MatMumpsSetIcntl(F, 14, 80);

        PetscRandomCreate(node.comm, &r);
        PetscRandomSetType(r, PETSCRAND);
        PetscRandomSetInterval(r, -1, 1);

        MatGetSize(K, &N, NULL);
        VecCreateMPI(node.comm, PETSC_DECIDE, N, &x);
        VecCreateMPI(node.comm, PETSC_DECIDE, N, &Sx);
        VecCreateMPI(node.comm, PETSC_DECIDE, N, &b);

        // initialize v (stored in Sx)
        VecSetRandom(x, r);
        VecCopy(x, Sx);
        VecAbs(Sx);
        VecPointwiseDivide(Sx, x, Sx);
        
        // x = abs( A' * Sx );
        // x = abs( K * (M \ Sx) );
        KSPSolve(ksp, Sx, b);
        MatMult(K, b, x);
        VecAbs(x);
        
        VecNorm(x, NORM_2, &e);
        VecScale(x, 1/e);
        e0 = 0;
        iter = 1;
        while (abs(e - e0) > opts.radtol()*e && iter <= opts.raditers())
        {
            e0 = e;
            // Sx = A * x
            // Sx = M \ (K * x)
            MatMult(K, x, b);
            KSPSolve(ksp, x, Sx);

            // x = A' * Sx
            // x = K * (M \ Sx)
            KSPSolve(ksp, Sx, b);
            MatMult(K, b, x);

            VecNorm(x, NORM_2, &normx);
            VecNorm(Sx, NORM_2, &normsx);
            e = normx / normsx;
            VecScale(x, 1/normx);
            
            iter++;
        }

        lR = 1.1*e;

        VecDestroy(&x);
        VecDestroy(&Sx);
        VecDestroy(&b);
        KSPDestroy(&ksp);
    }
    MPI_Bcast(&lR, 1, MPIU_REAL, 0, PETSC_COMM_WORLD);

    opts.set_R(lR);

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

    if (opts.totalsubproblems() == 1)
    {
        nleaves = 1;
        leafintervals[0] = splitintervals[0];
        leafintervals[1] = splitintervals[1];
    }

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
    PetscInt neval;
    PetscInt in_neval;

    neval = solveSubProblem(&intervals[node.id * 2 * opts.subproblemsperevaluator()], job0);

    std::vector<PetscInt> eval_counts(node.nevaluators);
    std::vector<PetscInt> eval_starts(node.nevaluators);
    for (int i = 0; i < node.nevaluators; i++)
    {
        if (node.id == i && node.rank == 0) in_neval = neval;
        MPI_Bcast(&in_neval, 1, MPIU_INT, i*node.size, PETSC_COMM_WORLD);
        eval_counts[i] = in_neval;
    }
    eval_starts[0] = 0;
    for (int i = 1; i < node.nevaluators; i++)
        eval_starts[i] = eval_starts[i-1] + eval_counts[i-1];

    node.neval = eval_counts[node.id];
    node.neval0 = eval_starts[node.id];

    if (node.worldrank == 0)
    {
        printf("Eigenvalues per node: [");
        for (int i = 0; i < node.nevaluators; i++)
            printf("%d ", eval_counts[i]);
        printf("]\n");

        printf("Eigenvalue starts per node: [");
        for (int i = 0; i < node.nevaluators; i++)
            printf("%d ", eval_starts[i]);
        printf("]\n");
    }
        

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

    Vec v;
    PetscInt lm, lm0, N;
    PetscReal lam, *v_data;

    MatGetSize(K, &N, NULL);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &v);
    VecGetLocalSize(v, &lm);
    MPI_Exscan(&lm, &lm0, 1, MPIU_INT, MPI_SUM, node.comm);
    if (node.rank == 0) lm0 = 0;

    std::vector<PetscInt> idxn(1);
    std::vector<PetscInt> idxm(lm);
    for (int i = 0; i < lm; i++)
        idxm[i] = lm0 + i;

    for (int i = 0; i < node.neval; i++)
    {
        idxn[0] = node.neval0 + i;
        EPSGetEigenpair(eps, i, &lam, NULL, v, NULL);
        VecGetArray(v, &v_data);
        MatSetValues(V, lm, &idxm[0], 1, &idxn[0], v_data, INSERT_VALUES);
        VecRestoreArray(v, &v_data);
        VecSetValue(lambda, idxn[0], lam, INSERT_VALUES);
    }
    MPI_Barrier(PETSC_COMM_WORLD);
    MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(lambda);
    VecAssemblyEnd(lambda);

    if(opts.save_eigenvalues())
    {
        PetscViewer viewer;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, opts.eigenvalues_filename(), 
            FILE_MODE_WRITE, &viewer);
        VecView(lambda, viewer);
        PetscViewerDestroy(&viewer);
    }

    if(opts.save_eigenbasis())
    {
        PetscViewer viewer;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, opts.eigenbasis_filename(), 
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
PetscInt eigen_mm::solveSubProblem(PetscReal *intervals, int job)
{
    MPI_Barrier(node.comm);
    double start_time = MPI_Wtime();

    if(opts.debug()) PetscPrintf(node.comm, "Solving interval [%lf, %lf]\n", 
        intervals[0], intervals[2*opts.subproblemsperevaluator()-1]);

    // Set up solver
    PetscInt nconv;

    if (opts.ksp_solver_type() == 0)
    {
        PetscOptionsInsertString(nullptr, "-st_type sinvert");
        PetscOptionsInsertString(nullptr, "-st_ksp_type preonly");
        PetscOptionsInsertString(nullptr, "-st_pc_type cholesky");
        PetscOptionsInsertString(nullptr, "-st_pc_factor_mat_solver_package mumps");
        PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_13 1");
        PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_14 80");
    }
    else if (opts.ksp_solver_type() == 1)
    {
        PetscOptionsInsertString(nullptr, "-st_type sinvert");
        PetscOptionsInsertString(nullptr, "-st_ksp_type chebyshev");
    }

    EPSCreate(node.comm,&eps);
    EPSSetOperators(eps,K,M);
    EPSSetProblemType(eps,EPS_GHEP);
    EPSSetFromOptions(eps);
    if (opts.eps_solver_type() == 0) 
    {
        std::vector<PetscReal> subint(opts.subproblemsperevaluator()+1);
        for (int i = 0; i < opts.subproblemsperevaluator(); i++)
            subint[i] = intervals[2*i];
        subint[opts.subproblemsperevaluator()] = intervals[2*opts.subproblemsperevaluator()-1];
        EPSSetWhichEigenpairs(eps, EPS_ALL);
        EPSSetInterval(eps, intervals[0], intervals[2*opts.subproblemsperevaluator()-1]);
        EPSKrylovSchurSetPartitions(eps, opts.subproblemsperevaluator());
        EPSKrylovSchurSetSubintervals(eps, &subint[0]);
        for (int i = 0; i < opts.subproblemsperevaluator()+1; i++)
            PetscPrintf(node.comm, "(%d) Subint[%d] = %lf\n", node.id, i, subint[i]);
    }

    // Solve
    EPSSolve(eps);

    // Process results
    EPSGetConverged(eps, &nconv); 

    node.neval += nconv;

    MPI_Barrier(node.comm);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse())
        PetscPrintf(node.comm, "%d %d %lf %lf %lf %d\n", node.id, job, elapsed, start_time, end_time, (int) nconv);
    else if (opts.debug())
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