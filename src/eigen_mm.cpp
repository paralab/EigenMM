#include "eigen_mm.h"

eigen_mm::eigen_mm() = default;

eigen_mm::~eigen_mm(){
    //MatDestroy(&K);
    //MatDestroy(&M);
};


int eigen_mm::init(Mat &K_in, Mat &M_in, SolverOptions opts_in)
{
    opts = opts_in;

    PetscPrintf(PETSC_COMM_WORLD, "Initializing communication hierarchy\n");

    // Initialize World Communicator
    MPI_Comm_rank(PETSC_COMM_WORLD, &(node.worldrank));
    MPI_Comm_size(PETSC_COMM_WORLD, &(node.worldsize));

    // Initialize Evaluator Communicator
    node.taskspernode = opts.taskspernode;
    node.processes_per_node = opts.nodesperevaluator * opts.taskspernode;
    node.id = node.worldrank / node.processes_per_node;
    node.size = node.worldsize / node.processes_per_node;
    MPI_Comm_split(PETSC_COMM_WORLD, node.id, node.worldrank, &(node.comm));
    MPI_Comm_rank(node.comm, &(node.rank));
    node.viewer = PETSC_VIEWER_STDOUT_(node.comm);

    // Initialize Row Communicator
    node.rowid = node.worldrank % node.processes_per_node;
    MPI_Comm_split(PETSC_COMM_WORLD, node.rowid, node.worldrank, &(node.rowcomm));
    MPI_Comm_rank(node.rowcomm, &(node.rowrank));
    MPI_Comm_size(node.rowcomm, &(node.rowsize)); 

    node.nevaluators = node.worldsize / node.processes_per_node;
    opts.nevaluators = node.nevaluators;
    opts.totalsubproblems = opts.nevaluators * opts.subproblemsperevaluator;

    PetscPrintf(PETSC_COMM_WORLD, "Distributing input system to evaluators\n");

    scatterInputMats(K_in, M_in);
}

int eigen_mm::solve(Mat *V_out, Vec *lambda_out)
{
    findUpperBound();
    formSubproblems();
    PetscInt neval = solveSubproblems();
//    formEigenbasis(neval);
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

void eigen_mm::findUpperBound()
{
    PetscReal lR;
    PetscInt rev, in_rev, iter;

    iter = 0;
    lR = (1 << node.id) * opts.R;
    rev = computeDev(lR, 3*lR, PETSC_TRUE);

    for (int i = 0; i < node.size; i++)
    {
        in_rev = rev;
        MPI_Bcast(&in_rev, 1, MPIU_INT, i*node.processes_per_node, PETSC_COMM_WORLD);
        if (in_rev <= 0)
        {
            lR = (1 << i) * opts.R;
            break;
        }
    }
    while(in_rev > 0)
    {
        iter++;
        lR = (1 << (node.id + iter*node.size)) * opts.R;
        rev = computeDev(lR, 3*lR, PETSC_TRUE);
        for (int i = 0; i < node.size; i++)
        {
            in_rev = rev;
            MPI_Bcast(&in_rev, 1, MPIU_REAL, i*node.processes_per_node, PETSC_COMM_WORLD);
            if(in_rev <= 0) 
            {
                lR = (1 << (i + iter*node.size)) * opts.R;
                break;
            }
        }
    }

    opts.R = lR;

    MPI_Barrier(PETSC_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Upper bound found: %lf\n", (double) lR);
}
void eigen_mm::formSubproblems()
{
    PetscReal leafintervals[10000];
    PetscReal splitintervals[10000];
    PetscInt  leafcounts[10000];

    PetscInt fleft, fright, noutput, split_index, leaf_index;
    PetscReal split;
    
    splitintervals[0] = opts.L;
    splitintervals[1] = opts.R;

    PetscPrintf(node.comm, "[Evaluator %d]: Total interval is [%lf, %lf], Nevt = %d, Totalsubproblems = %d\n", node.id, (double) opts.L, (double) opts.R, (int) opts.nevt, (int) opts.totalsubproblems);

    PetscInt nleaves = 0;
    int current_split = 1;
    int split_max_depth = 20;
    int job0 = node.id;
    int jobstride = node.size;
    int njobs = 1;

    while (nleaves       < opts.totalsubproblems && 
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
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 0], 1, MPIU_REAL, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 1], 1, MPIU_REAL, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 2], 1, MPIU_REAL, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
            MPI_Bcast(&leafintervals[2*nleaves + 4*job + 3], 1, MPIU_REAL, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
            MPI_Bcast(&leafcounts[nleaves + 2*job + 0],    1, MPIU_INT, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
            MPI_Bcast(&leafcounts[nleaves + 2*job + 1],    1, MPIU_INT, (job % node.size)*node.processes_per_node, PETSC_COMM_WORLD);
        }
        MPI_Barrier(PETSC_COMM_WORLD);

        noutput     = 2*njobs;
        split_index = 0;
        leaf_index  = 0;
        for (int i = 0; i < noutput; i++)
        {
            if (leafcounts[nleaves + i] > opts.nevt &&
                nleaves + noutput < opts.totalsubproblems &&
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
    PetscPrintf(PETSC_COMM_WORLD, "Subproblems formed : %d subproblems\n", (int) nleaves);
}
PetscInt eigen_mm::solveSubproblems()
{
    int job0       = node.id;
    int jobstride  = node.size;
    int njobs      = intervals.size() / 2;
    PetscInt neval = 0;
    PetscInt in_neval;
    for (int job = job0; job < njobs; job += jobstride)
    {
        neval += solveSubProblem(intervals[2*job+0], intervals[2*job+1]);
    }
    for (int i = 1; i < node.size; i++)
    {
        in_neval = neval;
        MPI_Bcast(&in_neval, 1, MPIU_INT, i*node.processes_per_node, PETSC_COMM_WORLD);
        if (node.id == 0) neval += in_neval;
    }
    MPI_Bcast(&neval, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    return neval;
}
void eigen_mm::formEigenbasis(PetscInt neval)
{
    MPI_Status status;
    MPI_Request request;

    int totalrowsize;
    std::vector<int> sizes(node.rowsize);
    std::vector<int> starts(node.rowsize);
    std::vector<PetscReal> rowdata;

    // Each row communicator broadcasts their data size
    sizes[node.rowrank] = lv_data.size();
    for (int i = 0; i < node.rowsize; i++)
        MPI_Bcast(&sizes[i], 1, MPI_INT, i, node.rowcomm);

    // Each row process determines all of the data starting 
    //  locations for its row
    starts[0] = 0;
    for (int i = 1; i < node.rowsize; i++)
        starts[i] = starts[i-1] + sizes[i-1];
    totalrowsize = starts[node.rowsize - 1] + sizes[node.rowsize - 1];

    // Each row communicator gathers data
    if (node.rowrank == 0) rowdata.resize(totalrowsize);
    MPI_Gatherv(&lv_data[0], lv_data.size(), MPIU_REAL, 
        &rowdata[0], &sizes[0], &starts[0], MPIU_REAL, 
        0, node.rowcomm);

    PetscInt N,m,n;
    MatGetSize(K, &N, NULL);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, neval, NULL, &V);
    MatGetLocalSize(V, &m, &n);
    std::vector<PetscReal> out_buffer(N);
    std::vector<PetscReal> in_buffer(N);

    for (int i = 0; i < node.nevaluators; i++)
    {
        if(node.rowrank == 0) 
        {
            for (int j = 0; j < rowdata.size()/m; j++)
                out_buffer[j] = lv_data[i + j*m];
            MPI_Isend(&out_buffer[0], N, MPIU_REAL, node.rowid*node.nevaluators+i, 
                i, PETSC_COMM_WORLD, &request);
        }
    }
    MPI_Recv(&in_buffer[0], N, MPIU_REAL, node.worldrank / node.nevaluators, 
        node.worldrank % node.nevaluators, PETSC_COMM_WORLD, &status);

    PetscScalar *V_data;
    MatDenseGetArray(V, &V_data);
    for (int j = 0; j < in_buffer.size(); j++)
        V_data[j] = in_buffer[j];
    MatDenseRestoreArray(V, &V_data);
}

void eigen_mm::scatterInputMats(Mat &K_in, Mat &M_in)
{
    PetscViewer viewer;

    const char *filenameK = "/scratch/kingspeak/serial/u0450449/fractional/matrices/tempK";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenameK, 
        FILE_MODE_WRITE, &viewer);
    MatView(K_in, viewer);
    PetscViewerDestroy(&viewer);

    const char *filenameM = "/scratch/kingspeak/serial/u0450449/fractional/matrices/tempM";
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filenameM, 
        FILE_MODE_WRITE, &viewer);
    MatView(M_in, viewer);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(node.comm, filenameK, 
        FILE_MODE_READ, &viewer);
    MatCreate(node.comm, &K);
    MatLoad(K, viewer);
    MatSetOption(K, MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);

    PetscViewerBinaryOpen(node.comm, filenameM, 
        FILE_MODE_READ, &viewer);
    MatCreate(node.comm, &M);
    MatLoad(M, viewer);
    MatSetOption(M, MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);
}
PetscInt eigen_mm::solveSubProblem(PetscReal a, PetscReal b)
{
    // Set up solver
    PetscInt nconv;
    EPS eps;
    char subinterval_string[1024] = "";
    sprintf(subinterval_string, "-eps_interval %lf,%lf", (double)a, (double)b);
    PetscOptionsInsertString(nullptr, subinterval_string);
    PetscOptionsInsertString(nullptr, "-st_type sinvert");
    PetscOptionsInsertString(nullptr, "-st_ksp_type preonly");
    PetscOptionsInsertString(nullptr, "-st_pc_type cholesky");
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

    PetscPrintf(node.comm, "[Evaluator %d]: Solved interval [%lf, %lf] and found %d eigenpairs.\n", node.id, (double) a, (double) b, (int) nconv);

    return nconv;
}
void eigen_mm::splitSubProblem(PetscReal a, PetscReal b, PetscReal *c, 
        PetscInt *out_ec_left, PetscInt *out_ec_right)
{
    PetscReal bleft, bright, split, ratio;
    PetscInt iter, ec_left, ec_right;

    PetscPrintf(node.comm, "[Evaluator %d]: Splitting interval [%lf, %lf]\n", node.id, (double) a, (double) b);

    bleft  = a;
    bright = b;
    iter   = 1;
    ratio  = 0.0;
    while (ratio < opts.splittol && iter < opts.splitmaxiters)
    {
        split = (bleft + bright) / 2.0;
        countInterval(a, split, &ec_left);
        countInterval(split, b, &ec_right);
        if (ec_left < ec_right) { bleft  = split; }
        else                    { bright = split; }
        ratio = fabs(std::min(ec_left, ec_right) / std::max(ec_left, ec_right));
        iter++;
    }
    c[0]            = split;
    out_ec_left[0]  = ec_left;
    out_ec_right[0] = ec_right;
    
    MPI_Barrier(node.comm);
    PetscPrintf(node.comm, "[Evaluator %d]: Split interval [%lf, %lf] into [%lf, %lf] (%d) and [%lf, %lf] (%d)\n",
        node.id, (double) a, (double) b, (double) a, (double) split, (int) ec_left, (double) split, (double) b, (int) ec_right);
}
PetscInt eigen_mm::computeDev(PetscReal a, PetscReal U, PetscBool rl)
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
    for (int k = 1; k <= opts.nv; k++)
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

        for (int j = 0; j <= opts.p; j++)
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
    retval = (PetscInt) round((PetscReal)1/(PetscReal)opts.nv * rev);
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
    while (abs(e - e0) > opts.radtol*e && iter <= opts.raditers)
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
void eigen_mm::countInterval(PetscReal a, PetscReal b, PetscInt *count)
{
    PetscInt reva = computeDev(a, opts.R, PETSC_TRUE);
    PetscInt revb = computeDev(b, opts.R, PETSC_TRUE);
    count[0] = reva - revb;
    MPI_Barrier(node.comm);
}

//double print_time(double t_start, double t_end, const std::string function_name, MPI_Comm comm){
//
//    int rank, nprocs;
//    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(comm, &nprocs);
//
//    double min, max, average;
//    double t_dif = t_end - t_start;
//
//    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
//    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
//    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
//    average /= nprocs;
//
//    if (rank==0)
//        std::cout << std::endl << function_name << "\nmin: " << min << "\nave: " << average << "\nmax: " << max << std::endl << std::endl;
//
//    return average;
//}