#include <mcompress.hpp>

int main ( int argc, char* argv[] )
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get parameters from arguments
    BFInt L;
    double tol;
    const char *input_filename  = nullptr;
    const char *output_filename = nullptr;
    BFInt verbose;
    if (argc < 1)
    {
        cout << "Missing required arguments: ./main [filename] (L = 3) (tol = 1e-4) (output_filename = '')\n";
        return 0;
    }
    else if (argc < 2)
    {
        input_filename = argv[1];
        L = 3;
        tol = 1e-4;
        verbose = 1;
    }
    else if (argc < 3)
    {
        input_filename = argv[1];
        L = atoi(argv[2]);
        tol = 1e-4;
        verbose = 1;
    }
    else if (argc < 4)
    {
        input_filename = argv[1];
        L = atoi(argv[2]);
        tol = atof(argv[3]);
        verbose = 1;
    }
    else if (argc < 5)
    {
        input_filename = argv[1];
        L = atoi(argv[2]);
        tol = atof(argv[3]);
        output_filename = argv[4];
        verbose = 1;
    }
    else
    {
        input_filename = argv[1];
        L = atoi(argv[2]);
        tol = atof(argv[3]);
        output_filename = argv[4];
        verbose = atoi(argv[5]);
    }

    // Prepare workspace and initialize butterfly permutation ids
    BFMat bfA(L);

    // Run and time encode phase
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = Clock::now();
    bf_encode(tol, input_filename, bfA, verbose);
    MPI_Barrier(MPI_COMM_WORLD);
    auto finish_time = Clock::now();

    // Mat *Aa0 = bfA.getAaPtr(0);
    // printf("Aa[0]: (%d by %d)\n", Aa0->getRows(), Aa0->getCols());

    // Get result statistics
    double elapsed = NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time-start_time).count();
    double compression_ratio = bfA.getCompressionRatio();
   
    // Display result statistics
    if (rank == 0)
    {
        // printf("Compression: %.4lf%%\n", compression_ratio*100.0);
        // printf("Elapsed time: %lf ms\n", elapsed);
        if (verbose == 0) printf("%.4lf %lf\n", compression_ratio*100.0, elapsed);
        else printf("compression ratio = %.4lf%% elapsed = %lf ms\n", compression_ratio*100.0, elapsed);
    }
    if (output_filename != nullptr)
        bf_save(output_filename, bfA);
}