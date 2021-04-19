#include <mcompress.hpp>

void computeStatistics( std::vector<double> vals, double *mean, double *var)
{
    double lmean = 0.0;
    double lvar = 0.0;
    for (BFInt i = 0; i < vals.size(); i++)
    {
        double lmean_update = lmean + (vals[i] - lmean)/(i+1);
        lvar = lvar + (vals[i] - lmean)*(vals[i] - lmean_update);
        lmean = lmean_update;
    }

    mean[0] = lmean;
    var[0] = lvar;
}

void random_unit_vector(Vec &x, BFInt rank)
{
    if (rank == 0)
    {
        std::random_device rng_device;
        std::default_random_engine rng_engine(rng_device());
        std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

        double nrm = 0.0;
        for (BFInt j = 0; j < x.size(); j++)
        {
            double v = uniform_dist(rng_engine);
            x[j] = v;
            nrm += v*v;
        }
        nrm = sqrt(nrm);
        
        for (BFInt j = 0; j < x.size(); j++)
        {
            x[j] /= nrm;
        }
    }
    MPI_Bcast(x.begin(), x.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void clear_vector(Vec &x)
{
    for (BFInt j = 0; j < x.size(); j++)
        x[j] = 0.0;
}

double compare_vecs(Vec &xc, Vec &xe)
{
    double nrm1 = 0.0;
    double nrm2 = 0.0;
    for (BFInt i = 0; i < xc.size(); i++)
    {
        nrm1 += (xc[i] - xe[i]) * (xc[i] - xe[i]);
        nrm2 += xe[i] * xe[i];
    }
    return sqrt(nrm1) / sqrt(nrm2);
}

void matvec_timing_experiment(BFInt nv, BFMat &bfA, Mat &A, int verbose)
{
    std::vector<double> elapsed(nv);
    std::vector<double> relerr(nv);
    std::vector<double> mult_time(nv);
    std::vector<double> wait_time(nv);
    std::vector<double> comm_time(nv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BFInt m = bfA.getGlobalRows();
    BFInt n = bfA.getGlobalCols();

    BFInt L = bfA.getL();
    BFInt P = 1 << L;
    std::vector<MPI_Request> recvRequests(2*P);
    std::vector<MPI_Request> sendRequests(2*P);
    std::vector<MPI_Status> recvStatuses(2*P);

    Vec x(n);
    Vec b(m);
    Vec be(m);

    for (BFInt i = 0; i < nv; i++)
    {
        random_unit_vector(x, rank);
        clear_vector(b);

        // each process obtains the portions of x for which they are reponsible
        for (BFInt j = 0; j < bfA.getNumberOwned(0); j++)
            std::copy(x.begin()+bfA.getX0(bfA.getOwned(0,j)/2), x.begin()+bfA.getX0(bfA.getOwned(0,j)/2+1), bfA.getXpData(j));

        // if (i % 100 == 0 && rank == 0) printf(" Starting matvec %d / %d\n", i+1, nv);
        MPI_Barrier(MPI_COMM_WORLD);
        auto start_time = Clock::now();

        //bfA.apply(x, b);
        bfA.timed_apply(x, b, &wait_time[i], &mult_time[i], &comm_time[i], true);

        MPI_Barrier(MPI_COMM_WORLD);
        auto stop_time = Clock::now();
        for (BFInt j = 0; j < bfA.getNumberOwned(L); j++)
        {
            MPI_Isend(bfA.getXpData(j), bfA.getAaRows(bfA.getOwned(L,j)), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendRequests[j]);
        }
        if (rank == 0)
        {
            for (BFInt j = 0; j < 2*P; j++)
                MPI_Irecv(b.begin() + bfA.getY0(j), bfA.getAaRows(j), MPI_DOUBLE,
                    bfA.getOwnedMap(L, j), 0, MPI_COMM_WORLD, &recvRequests[j]);
            for (BFInt j = 0; j < 2*P; j++)
                MPI_Wait(&recvRequests[j], &recvStatuses[j]);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) bf_matvec(A, x, be);
        if (rank == 0) relerr[i] = compare_vecs(b, be);

        elapsed[i] = NS2MS * 
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double elapsed_mean = 0.0; double elapsed_var = 0.0;
    double relerr_mean = 0.0;  double relerr_var = 0.0;
    double multtime_mean = 0.0; double multtime_var = 0.0;
    double waittime_mean = 0.0; double waittime_var = 0.0;
    double commtime_mean = 0.0; double commtime_var = 0.0;
    if (rank == 0)
    {
        computeStatistics(elapsed,   &elapsed_mean,  &elapsed_var);
        computeStatistics(relerr,    &relerr_mean,   &relerr_var);
        computeStatistics(mult_time, &multtime_mean, &multtime_var);
        computeStatistics(wait_time, &waittime_mean, &waittime_var);
        computeStatistics(comm_time, &commtime_mean, &commtime_var);
        if (verbose == 0) printf("%d %d %lf %d %lf %lf %lf %lf %lf %lf\n", bfA.getL(), 1 << bfA.getL(), bfA.getInputTolerance(), n, bfA.getCompressionRatio(), elapsed_mean, relerr_mean, multtime_mean, waittime_mean, commtime_mean);
        else printf("L = %d, P = %d, eps = %lf, N = %d, compressionRatio = %lf, average elapsed = %lf, average accuracy = %lf, mult time = %lf, wait time = %lf, comm_time = %lf\n", bfA.getL(), 1 << bfA.getL(), bfA.getInputTolerance(), n, bfA.getCompressionRatio(), elapsed_mean, relerr_mean, multtime_mean, waittime_mean, commtime_mean);
    }
}

int main ( int argc, char* argv[] )
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get parameters from arguments
    BFInt nv;
    const char *input_filename  = nullptr;
    const char* exact_filename = nullptr;
    BFInt verbose;
    if (argc < 1)
    {
        cout << "Missing required arguments: ./main [filename] (nv = 100)\n";
        return 0;
    }
    else if (argc < 2)
    {
        input_filename = argv[1];
        nv = 100;
        verbose = 1;
    }
    else if (argc < 3)
    {
        input_filename = argv[1];
        nv = atoi(argv[2]);
        verbose = 1;
    }
    else if (argc < 4)
    {
        input_filename = argv[1];
        nv = atoi(argv[2]);
        exact_filename = argv[3];
        verbose = 1;
    }
    else
    {
        input_filename = argv[1];
        nv = atoi(argv[2]);
        exact_filename = argv[3];
        verbose = atoi(argv[4]);
    }

    // Prepare workspace and initialize butterfly permutation ids
    BFMat bfA;
    // if (rank == 0) printf("Loading bfA matrix\n");
    bf_load(input_filename, bfA);

    // if (rank == 0) printf("Loading exact input matrix\n");
    BFInt nRows, nCols;
    std::vector<double> Adata;
    if (rank == 0) petsc_binary_read(exact_filename, Adata, &nRows, &nCols);
    else { nRows = 1; nCols = 1; Adata.resize(1); }
    Mat A_in(nRows, nCols, Adata);

    // Run and time encode phase
    // if (rank == 0) printf("Beginning matvec timing experiment\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = Clock::now();
    matvec_timing_experiment(nv, bfA, A_in, verbose);
    MPI_Barrier(MPI_COMM_WORLD);
    auto finish_time = Clock::now();

    // Get result statistics
    double elapsed = NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time-start_time).count();
    double compression_ratio = bfA.getCompressionRatio();
   
    // Display result statistics
    if (rank == 0)
    {
        // printf("Compression: %.4lf%%\n", compression_ratio*100.0);
        // printf("Total elapsed time: %lf ms\n", elapsed);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}