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

int eigen_mm::init(Mat &K_in, Mat &M_in, SolverOptions &opts_in)
{
    opts = opts_in;

    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    MPI_Comm_rank(PETSC_COMM_WORLD, &(node.worldrank));
    MPI_Comm_size(PETSC_COMM_WORLD, &(node.worldsize));

    int count = 0;
    int resultlen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    uint32_t processor_id;
    uint32_t evaluator_id;
    std::vector<uint32_t> processor_ids;
    std::vector<uint32_t> evaluator_ids;
    std::map<uint32_t, uint32_t> processor_to_evaluator;

    MPI_Get_processor_name(processor_name, &resultlen);
    processor_id = adler32((unsigned char *) &processor_name[0], resultlen*sizeof(char));
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

    // multiply evaluator_id by the number of subproblems per evaluator
    // determine number of processors per subproblem
    // n_id = sppe * e_id + floor(e_rank / sppn)

    MPI_Comm nodecomm;
    int nodecomm_size, nodecomm_rank;
    MPI_Comm_split(PETSC_COMM_WORLD, evaluator_id, node.worldrank, &nodecomm);
    MPI_Comm_size(nodecomm, &nodecomm_size);
    MPI_Comm_rank(nodecomm, &nodecomm_rank);

    int sppn = nodecomm_size / opts.subproblemsperevaluator();
    evaluator_id = evaluator_id * opts.subproblemsperevaluator() + nodecomm_rank / sppn;

    MPI_Comm_split(PETSC_COMM_WORLD, evaluator_id, node.worldrank, &(node.comm));
    MPI_Comm_size(node.comm, &(node.size));
    MPI_Comm_rank(node.comm, &(node.rank));
    node.id = evaluator_id;

    node.nevaluators = opts.subproblemsperevaluator() * count / opts.nodesperevaluator();
    opts.set_nevaluators(node.nevaluators);
    opts.set_totalsubproblems(opts.nevaluators());
    node.neval = 0;

    MatCreateRedundantMatrix(K_in, node.nevaluators, node.comm, MAT_INITIAL_MATRIX, &K);
    MatCreateRedundantMatrix(M_in, node.nevaluators, node.comm, MAT_INITIAL_MATRIX, &M);

    M_global = M_in;
    K_global = K_in;

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report init time
    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(init) Elapsed: %lf\n", elapsed);
}

int eigen_mm::solve(Mat *V_out, Vec *lambda_out)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    if (opts.R() <= opts.L() && opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "finding upper bound\n");
    if (opts.R() <= opts.L()) findUpperBound();
    
    if (opts.nevals() > 0 && opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "rescaling interval to contain first %d eigenvalues\n", opts.nevals());
    if (opts.nevals() > 0) rescaleInterval();
    
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming subproblems\n");
    formSubproblems();
    
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "solving subproblems\n");
    PetscInt neval = solveSubproblems();
    
    if (opts.debug()) PetscPrintf(PETSC_COMM_WORLD, "forming eigenbasis\n");
    formEigenbasis(neval);

    *V_out = V;
    *lambda_out = lambda;

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(solve) Elapsed: %lf, Eigenvalues found: %d\n", elapsed, (int) neval);
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
        MatMumpsSetIcntl(F, 14, 140);

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
    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf\n", elapsed, start_time, end_time, (double) lR);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(findUpperBound) Elapsed: %lf, Upper Bound: %lf\n", elapsed, (double) lR);
}
void eigen_mm::rescaleInterval()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    PetscReal lR;

    if (node.id == 0)
    {
        PetscReal bleft, bright, split, err;
        PetscInt iter, ec, N;

        MatGetSize(K, &N, NULL);

        bleft  = opts.L();
        bright = opts.R();
        iter   = 1;
        err    = (PetscReal) N;
        while (abs(err) > 10 & iter < 20)
        {
            split = (bleft + bright)/2.0;
            countInterval(opts.L(), split, &ec);
            err = ec - opts.nevals();
            if (err > 0)
                bright = split;
            else
                bleft = split;
            iter++;
        }

        lR = split;
    }
    MPI_Bcast(&lR, 1, MPIU_REAL, 0, PETSC_COMM_WORLD);

    opts.set_R(lR);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report findUpperBound timing
    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf\n", elapsed, start_time, end_time, (double) lR);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(rescaleInterval) Elapsed: %lf, Upper Bound: %lf\n", elapsed, (double) lR);
}
void eigen_mm::formSubproblems()
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    PetscInt N;
    MatGetSize(K, &N, NULL);

    PetscInt n = node.nevaluators;

    PetscInt Nbar, Nhat;
    PetscReal err;
    std::vector<PetscInt> rev(n+1);
    std::vector<PetscInt> C(n);
    std::vector<PetscReal> x(n+1);

    PetscReal bestCerr = N;
    std::vector<PetscInt> bestRev(n+1);
    std::vector<PetscInt> bestC(n);
    std::vector<PetscReal> bestX(n+1);

    if (n > 1)
    {
        // (root node) Initialize parameters and x, broadcast to all processes
        if (node.id == 0)
        {
            rev[0] = N;
            rev[n] = (opts.nevals() > 0) ? (N - opts.nevals()) : 0;
            Nbar = rev[0] - rev[n];
            Nhat = Nbar / n;

            x[0] = opts.L();
            x[n] = opts.R();
            PetscReal step = (x[n] - x[0])/ (PetscReal) n;
            for (int i = 1; i < n; i++)
                x[i] = x[0] + i*step;
        }
        MPI_Bcast(&Nbar,   1,   MPIU_INT,  0, PETSC_COMM_WORLD);
        MPI_Bcast(&Nhat,   1,   MPIU_INT,  0, PETSC_COMM_WORLD);
        MPI_Bcast(&rev[0], 1,   MPIU_INT,  0, PETSC_COMM_WORLD);
        MPI_Bcast(&rev[n], 1,   MPIU_INT,  0, PETSC_COMM_WORLD);
        MPI_Bcast(&x[0],   n+1, MPIU_REAL, 0, PETSC_COMM_WORLD);

        // stage 1: global refinement
        for (int k = 0; k < opts.nk(); k++)
        {
            if (node.id > 0) rev[node.id] = computeDev_exact(x[node.id], PETSC_TRUE);
            
            for (int i = 1; i < n; i++)
                MPI_Bcast(&rev[i], 1, MPIU_INT, i*node.size, PETSC_COMM_WORLD);

            if (node.id == 0)
            {
                err = 0.0;
                for (int i = 0; i < n; i++)
                {
                    C[i] = rev[i] - rev[i+1];
                    err = std::max((PetscReal) abs(C[i] - Nhat), err);
                }
                err = err / (PetscReal) Nbar;
            }
            MPI_Bcast(&err,      1, MPIU_REAL, 0, PETSC_COMM_WORLD);
            MPI_Bcast(&rev[0], n+1, MPIU_INT,  0, PETSC_COMM_WORLD);
            MPI_Bcast(&C[0],     n, MPIU_INT,  0, PETSC_COMM_WORLD);

            // update bests
            if (err < bestCerr)
            {
                bestX = x;
                bestRev = rev;
                bestC = C;
                bestCerr = err;
            }

            if (k < opts.nk()-1)
            {
                // global refinement step
                if (node.worldrank == 0) global_refine(n, x, C, Nhat);
                MPI_Bcast(&x[0],   n+1, MPIU_REAL, 0, PETSC_COMM_WORLD);
            }
            else
            {
                // set data equal to the best observed
                x = bestX;
                rev = bestRev;
                C = bestC;
                err = bestCerr;
            }
            
        }

        // stage 2: local refinement
        for (int b = 0; b < opts.nb(); b++)
        {
            // even pass
            if (node.id % 2 == 0 && node.id < n-1)
            {
                // determine if interval pair can be balanced
                PetscReal ratio = (std::max(C[node.id], C[node.id+1]) != 0) 
                                ?  std::min(C[node.id], C[node.id+1]) 
                                /  std::max(C[node.id], C[node.id+1]) 
                                :  0.0;
                bool cond1 = C[node.id] > Nhat && C[node.id+1] < Nhat;
                bool cond2 = C[node.id] < Nhat && C[node.id+1] > Nhat;
                bool cond  = cond1 || cond2;


                // if valid, balance interval pair
                if (ratio < opts.splittol() && cond)
                {
                    balance_intervals(  x[node.id],   x[node.id+2],   &x[node.id+1], 
                                      rev[node.id], rev[node.id+2], &rev[node.id+1],
                                       &C[node.id],  &C[node.id+1]);
                }
            }
            MPI_Barrier(PETSC_COMM_WORLD);
            for (int i = 0; i < n; i+=2)
            {
                MPI_Bcast(&x[i+1],   1, MPIU_REAL, i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&rev[i+1], 1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&C[i],     1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&C[i+1],   1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
            }

            // odd pass
            if (node.id % 2 == 1 && node.id < n-1)
            {
                // determine if interval pair can be balanced
                PetscReal ratio = (std::max(C[node.id], C[node.id+1]) != 0) 
                                ?  std::min(C[node.id], C[node.id+1]) 
                                /  std::max(C[node.id], C[node.id+1]) 
                                :  0.0;
                bool cond1 = C[node.id] > Nhat && C[node.id+1] < Nhat;
                bool cond2 = C[node.id] < Nhat && C[node.id+1] > Nhat;
                bool cond  = cond1 || cond2;


                // if valid, balance interval pair
                if (ratio < opts.splittol() && cond)
                {
                    balance_intervals(  x[node.id],   x[node.id+2],   &x[node.id+1], 
                                      rev[node.id], rev[node.id+2], &rev[node.id+1],
                                       &C[node.id],  &C[node.id+1]);
                }
            }
            MPI_Barrier(PETSC_COMM_WORLD);
            for (int i = 1; i < n; i+=2)
            {
                MPI_Bcast(&x[i+1],   1, MPIU_REAL, i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&rev[i+1], 1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&C[i],     1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
                MPI_Bcast(&C[i+1],   1, MPIU_INT,  i*node.size, PETSC_COMM_WORLD);
            }
        }
        
        for (int i = 0; i < n; i++)
        {
            intervals.push_back(x[i]);
            intervals.push_back(x[i+1]);
        }
    }
    else
    {
        intervals.push_back(opts.L());
        intervals.push_back(opts.R());
    }


    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    // report formSubproblems timing
    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %d\n", elapsed, start_time, end_time, (int) n);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(formSubproblems) Elapsed: %lf, Total Subproblems: %d\n", elapsed, (int) n);
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
    PetscInt retval = 0;

    neval = solveSubProblem(&intervals[2 * node.id], job0);

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

    for (int i = 0; i < node.nevaluators; i++)
        retval += eval_counts[i];

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %d\n", elapsed, start_time, end_time, (int) retval);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(solveSubproblems) Elapsed: %lf, Total Eigenpairs Found: %d\n", elapsed, (int) retval);

    return retval;
}
void eigen_mm::formEigenbasis(PetscInt neval)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    std::vector<Vec> v;
    Vec Mv, v0;
    Vec t1, t2;
    PetscInt lm, lm0, N;
    PetscReal nrm, lam, *v_data, dot1, dot2;

    MatGetSize(K, &N, NULL);
    MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, neval, NULL, &V);
    MatSetUp(V);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, neval, &lambda);

    VecCreateMPI(node.comm, PETSC_DECIDE, N, &v0);
    v.push_back(v0);

    VecCreateMPI(node.comm, PETSC_DECIDE, N, &Mv);
    VecGetLocalSize(v[0], &lm);
    MPI_Exscan(&lm, &lm0, 1, MPIU_INT, MPI_SUM, node.comm);
    if (node.rank == 0) lm0 = 0;

    std::vector<PetscInt> idxn(1);
    std::vector<PetscInt> idxm(lm);
    for (int i = 0; i < lm; i++)
        idxm[i] = lm0 + i;

    VecCreateMPI(node.comm, PETSC_DECIDE, N, &t1);
    VecCreateMPI(node.comm, PETSC_DECIDE, N, &t2);

    int i = 0;
    int k = 0;
    EPSGetEigenpair(eps, i, &lam, NULL, v[0], NULL);
    VecSetValue(lambda, node.neval0, lam, INSERT_VALUES);
    while (i < node.neval)
    {
        idxn[0] = node.neval0 + i;

        // Normalize v[0] with respect to M (v' * M * v = nrm)
        MatMult(M, v[0], Mv);
        VecDot(v[0], Mv, &nrm);
        VecScale(v[0], 1.0/sqrt(nrm));

        // Get vectors from EPS until v[k] is orthogonal to v[0] 
        //  or there are no more vectors
        k = 0;
        nrm = 1.0;

        while (i+k < node.neval-1 && nrm > 1e-5)
        {
            k++;
            if (k >= v.size())
            {
                Vec vk;
                VecCreateMPI(node.comm, PETSC_DECIDE, N, &vk);
                v.push_back(vk);
            }
            EPSGetEigenpair(eps, i+k, &lam, NULL, v[k], NULL);
            VecSetValue(lambda, idxn[0]+k, lam, INSERT_VALUES);
            VecDot(v[k], Mv, &nrm);
        }

        // (MGS) Orthogonalize v[0] through v[k-1] with respect to M
        if (k > 0)
        {
            // v[0] is normalized
            for (int j = 1; j < k; j++)
            {
                for (int jj = 0; jj < j; jj++)
                {
                    // Mv = M*v[jj]
                    MatMult(M, v[jj], Mv);

                    // dot1 = dot(v[j], Mv)
                    VecDot(v[j], Mv, &dot1);

                    // dot2 = dot(v[jj], Mv)
                    VecDot(v[jj], Mv, &dot2);

                    // v[j] -= dot1/dot2 * v[jj]
                    VecAXPY(v[j], -dot1/dot2, v[jj]);
                }

                // normalize v[j]
                MatMult(M, v[j], Mv);
                VecDot(v[j], Mv, &nrm);
                VecScale(v[j], 1.0/sqrt(nrm));
            }
        }

        if (k > 0)
        {
            // Insert vectors v[0] through v[k-1] into V
            for (int j = 0; j < k; j++)
            {
                VecGetArray(v[j], &v_data);
                MatSetValues(V, lm, &idxm[0], 1, &idxn[0]+j, v_data, INSERT_VALUES);
                VecRestoreArray(v[j], &v_data);
            }

            // Copy v[k] into v[0]
            VecCopy(v[k], v[0]);

            // Increment i
            i += k;
        }
        else
        {
            VecGetArray(v[0], &v_data);
            MatSetValues(V, lm, &idxm[0], 1, &idxn[0], v_data, INSERT_VALUES);
            VecRestoreArray(v[0], &v_data);
            i++;
        }
    }

    if (opts.save_correctness())
    {
        for (int j = 0; j < node.neval; j++)
        {
            PetscReal residual;

            // K * vk - lambdak * M * vk
            MatMult(K, v[0], t1);
            MatMult(M, v[0], t2);
            VecAXPY(t1, -lam, t2);
            VecNorm(t1, NORM_2, &residual);

            residuals.push_back(residual);
        }
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

    if (opts.save_correctness() && node.rank == 0)
    {
        char correctness_filename[2048];
        sprintf(correctness_filename, "%s_correctness_%d", opts.correctness_filename(), node.id);
        std::ofstream output_file(correctness_filename);
        std::ostream_iterator<PetscReal> output_iterator(output_file, "\n");
        std::copy(residuals.begin(), residuals.end(), output_iterator);
    }

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    //VecDestroy(&v);
    VecDestroy(&Mv);
    VecDestroy(&t1);
    VecDestroy(&t2);
    EPSDestroy(&eps);

    // // Temporary orthogonality check
    // // =============================================================
    // PetscPrintf(PETSC_COMM_WORLD, "Beginning correctness check\n");

    // // Want to verify that the following conditions are satisfied
    // // V'*M*V = I        [ norm(V'*M*V - I) ]
    // Mat temp1, temp2;
    // MatMatMult(M_global, V, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp1);
    // MatTransposeMatMult(V, temp1, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &temp2);
    // MatShift(temp2, -1.0);
    // MatNorm(temp2, NORM_FROBENIUS, &nrm);
    // PetscPrintf(PETSC_COMM_WORLD, "norm(V'*M*V - I) = %.16lf\n", nrm);

    // // V'*K*V = lambda   [ norm(V'*K*V - lambda) / norm(lambda) ]
    // MatMatMult(K_global, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &temp1);
    // MatTransposeMatMult(V, temp1, MAT_REUSE_MATRIX, PETSC_DEFAULT, &temp2);
    // VecNorm(lambda, NORM_2, &lam);
    // VecScale(lambda, -1.0);
    // MatDiagonalSet(temp2, lambda, ADD_VALUES);
    // VecScale(lambda, -1.0);
    // MatNorm(temp2, NORM_FROBENIUS, &nrm);
    // PetscPrintf(PETSC_COMM_WORLD, "norm(V'*K*V - lambda) / norm(lambda) = %.16lf\n", nrm/lam);

    // MatDestroy(&temp1);
    // MatDestroy(&temp2);
    // // =============================================================

    for (int j = 0; j < v.size(); j++)
        VecDestroy(&v[j]);

    if (opts.terse() && opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf\n", elapsed, start_time, end_time);
    else if (opts.debug())
        PetscPrintf(PETSC_COMM_WORLD, "(formEigenbasis) Elapsed: %lf\n", elapsed);
}
// ================================================

// ==================== Evaluator ====================
PetscInt eigen_mm::solveSubProblem(PetscReal *intervals, int job)
{
    MPI_Barrier(node.comm);
    double start_time = MPI_Wtime();

    if(opts.debug()) PetscPrintf(node.comm, "[Node %d]: Solving interval [%lf, %lf]\n", 
        node.id, intervals[0], intervals[1]);

    // Set up solver
    PetscInt nconv;

    if (opts.ksp_solver_type() == 0)
    {
        PetscOptionsInsertString(nullptr, "-st_type sinvert");
        PetscOptionsInsertString(nullptr, "-st_ksp_type preonly");
        PetscOptionsInsertString(nullptr, "-st_pc_type cholesky");
        PetscOptionsInsertString(nullptr, "-st_pc_factor_mat_solver_package mumps");
        PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_13 1");
        PetscOptionsInsertString(nullptr, "-mat_mumps_icntl_14 140");
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
        EPSSetWhichEigenpairs(eps, EPS_ALL);
        EPSSetInterval(eps, intervals[0], intervals[1]);
        EPSKrylovSchurSetPartitions(eps, 1);
        EPSKrylovSchurSetSubintervals(eps, intervals);
    }

    EPSSolve(eps);

    EPSGetConverged(eps, &nconv); 

    node.neval += nconv;

    MPI_Barrier(node.comm);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (opts.debug() && opts.terse())
        PetscPrintf(node.comm, "%d %d %lf %lf %lf %d\n", node.id, job, elapsed, start_time, end_time, (int) nconv);
    else if (opts.debug())
        PetscPrintf(node.comm, "(solveSubproblem) Evaluator: %d, Subinterval: %d, Elapsed: %lf, Number of Eigenpairs: %d\n", node.id, job, elapsed, (int) nconv);

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

    if (opts.terse() && opts.debug())
        PetscPrintf(node.comm, "%d %lf %lf %lf %d %d %lf %lf %lf", node.id, elapsed, start_time, end_time, (int) ec_left, (int) ec_right, (double) a, (double) b, (double) split);
    else if (opts.debug())
        PetscPrintf(node.comm, "(splitSubproblem) Evaluator: %d, Elapsed: %lf, Left Count: %d, Right Count: %d, a = %lf, b = %lf, c = %lf\n", node.id, elapsed, (int) ec_left, (int) ec_right, (double) a, (double) b, (double) split);
}

void eigen_mm::global_refine(PetscInt n,
                             std::vector<PetscReal> &x, 
                             std::vector<PetscInt>   C, 
                             PetscInt Nhat)
{
    std::vector<PetscReal> x2(n+1);
    x2[0] = x[0];
    x2[n] = x[n];

    PetscReal xi, xj, xjm1, xp, a;
    PetscInt tempcount, prevcount, count;
    PetscInt j = 0;

    xi = x[0];
    for (int i = 0; i < n-1; i++)
    {
        while (j <= n && x[j] <= xi) { j++; }
        xj   = x[j];
        xjm1 = x[j-1];

        tempcount = ((xj - xi) / (xj - xjm1) * C[j-1]);
        count = tempcount;

        xp = -1.0;
        prevcount = 0;
        while (xp < 0)
        {
            if (count > Nhat)
            {
                a = ((PetscReal) (Nhat - prevcount)) / (PetscReal) (count - prevcount);
                xp = xi + a * (xj - xi);
            }
            else if (count < Nhat)
            {
                prevcount = prevcount + tempcount;
                xi = xj;
                xjm1 = x[j];
                xj = x[j+1];
                j++;
                tempcount = ((xj - xi) / (xj - xjm1) * C[j-1]);
                count = prevcount + tempcount;
            }
            else
            {
                xp = xj;
            }
        }
        x2[i+1] = xp;
        xi      = xp;
    }

    x = x2;
}
void eigen_mm::balance_intervals(PetscReal   a, PetscReal   b, PetscReal   *c, 
                                 PetscInt reva, PetscInt revb, PetscInt *revc,
                                 PetscInt  *Cl, PetscInt  *Cr)
{
    PetscInt iter = 1;
    PetscReal C_a_c = reva - *revc;
    PetscReal C_c_b = *revc - revb;
    PetscReal L, R, ratio;
    if (C_a_c < C_c_b)
    {
        L = *c;
        R =  b;
    }
    else
    {
        L =  a;
        R = *c;
    }
    ratio = (std::max(C_a_c, C_c_b) > 0) 
          ?  std::min(C_a_c, C_c_b) 
          /  std::max(C_a_c, C_c_b)
          :  0.0;

    PetscInt  bestsplit = *c;
    PetscReal bestratio = ratio;
    PetscInt  bestrev = *revc;

    PetscReal split;
    PetscInt  revsplit;
    PetscInt  C_a_split;
    PetscInt  C_split_b;
    while (ratio < opts.splittol() && iter < 10)
    {
        split = (L+R)/2.0;
        revsplit = computeDev_exact(split, PETSC_TRUE);
        C_a_split = reva - revsplit;
        C_split_b = revsplit - revb;
        if (C_a_split < C_split_b)
            L = split;
        else
            R = split;
        ratio = (std::max(C_a_split, C_split_b) > 0) 
              ?  std::min(C_a_split, C_split_b) 
              /  std::max(C_a_split, C_split_b)
              :  0.0;
        iter++;

        if (ratio > bestratio)
        {
            bestsplit = split;
            bestrev   = revsplit;
            bestratio = ratio;
        }
    }

    *c = bestsplit;
    *revc = bestrev;
    *Cl = reva - bestrev;
    *Cr = bestrev - revb;
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
    MatMumpsSetIcntl(L, 14, 140);
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


void eigen_mm::exactEigenvalues_square_neumann(int Ne, 
    std::vector<PetscReal> &lambda, 
    std::vector<PetscReal> &eta1, 
    std::vector<PetscReal> &eta2)
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int count = 0;
    int d = ceil(sqrt(Ne));
    lambda.resize(Ne);
    eta1.resize(Ne);
    eta2.resize(Ne);

    if (rank == 0)
    {
        // Generate set of eigenvalues
        std::vector<std::pair<PetscReal,int>> lambda_local;
        std::vector<PetscReal> eta1_local;
        std::vector<PetscReal> eta2_local;
        PetscReal L, E1, E2;
        for (int i = 0; i < d+1; i++)
        {
            E1 = (i*PI) * (i*PI);
            for (int j = 0; j < d+1; j++)
            {
                E2 = (j*PI) * (j*PI);
                L = E1 + E2;

                if (L > 0.0)
                {
                    std::pair<PetscReal,int> p(L,count);
                    lambda_local.push_back(p);
                    eta1_local.push_back(E1);
                    eta2_local.push_back(E2);
                    count++;
                }
            }
        }

        // Sort lambda from smallest to largest
        std::sort(lambda_local.begin(), lambda_local.end());
        
        // Output first Ne smallest eigenvalues
        for (int i = 0; i < Ne; i++)
        {
            lambda[i] = lambda_local[i].first;
            eta1[i] = eta1_local[lambda_local[i].second];
            eta2[i] = eta2_local[lambda_local[i].second];
        }
    }

    // Broadcast results to all processes
    MPI_Bcast(&lambda[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&eta1[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&eta2[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
}
void eigen_mm::exactEigenvalues_cube_neumann(int Ne, 
    std::vector<PetscReal> &lambda, 
    std::vector<PetscReal> &eta1, 
    std::vector<PetscReal> &eta2,
    std::vector<PetscReal> &eta3)
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    int count = 0;
    int d = ceil(pow(Ne,1.0/3.0));
    lambda.resize(Ne);
    eta1.resize(Ne);
    eta2.resize(Ne);
    eta3.resize(Ne);

    if (rank == 0)
    {
        // Generate set of eigenvalues
        std::vector<std::pair<PetscReal,int>> lambda_local;
        std::vector<PetscReal> eta1_local;
        std::vector<PetscReal> eta2_local;
        std::vector<PetscReal> eta3_local;
        PetscReal L, E1, E2, E3;
        for (int i = 0; i < d+1; i++)
        {
            E1 = (i*PI) * (i*PI);
            for (int j = 0; j < d+1; j++)
            {
                E2 = (j*PI) * (j*PI);
                for (int k = 0; k < d+1; k++)
                {
                    E3 = (k*PI) * (k*PI);
                    L = E1 + E2 + E3;

                    if (L > 0.0)
                    {
                        std::pair<PetscReal,int> p(L,count);
                        lambda_local.push_back(p);
                        eta1_local.push_back(E1);
                        eta2_local.push_back(E2);
                        eta3_local.push_back(E3);
                        count++;
                    }
                }
                
            }
        }

        // Sort lambda from smallest to largest
        std::sort(lambda_local.begin(), lambda_local.end());
        
        // Output first Ne smallest eigenvalues
        for (int i = 0; i < Ne; i++)
        {
            lambda[i] = lambda_local[i].first;
            eta1[i] = eta1_local[lambda_local[i].second];
            eta2[i] = eta2_local[lambda_local[i].second];
            eta3[i] = eta3_local[lambda_local[i].second];
        }
    }

    // Broadcast results to all processes
    MPI_Bcast(&lambda[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&eta1[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&eta2[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&eta3[0], Ne, MPIU_REAL, 0, PETSC_COMM_WORLD);
}