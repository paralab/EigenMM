#include "eigen_mm.h"
#include <zfp.h>

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems);
void checkCorrectness(Mat K, Mat M, Mat V, Vec lambda, PetscReal *norms, SolverOptions options);
void checkOrthogonality(Mat M, Mat V, SolverOptions options);
void checkCompress1(Mat V, double *compress_times, double *decompress_times, double *data_reduction, SolverOptions options);
void checkCompress2(Mat V, double *compress_times, double *decompress_times, double *data_reduction, SolverOptions options);
double compress1D(double* array, int nx, double tolerance, int decompress);
double compress2D(double* array, int nx, int ny, double tolerance, int decompress);

int main(int argc, char *argv[])
{
    Mat K, M, V;
    Vec lambda;
    SolverOptions options;
    eigen_mm solver;

    // Set up solver paramters
    options.set_p(0);
    options.set_subproblemsperevaluator(1);
    //options.set_saveV(true, "/uufs/chpc.utah.edu/common/home/u0450449/Fractional/EigenMM/cube/");
    options.set_savelambda(true, "/uufs/chpc.utah.edu/common/home/u0450449/Fractional/EigenMM/cube/");
    options.set_savecorrectness(true, "/uufs/chpc.utah.edu/common/home/u0450449/Fractional/EigenMM/correctness/");
    options.set_debug(true);

    SlepcInitialize(&argc,&argv,NULL,NULL);

    // geometry paramters
    PetscInt dim, nelems, order;
    char dtype[1024];
    char output_filepath[1024];
    PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL);
    PetscOptionsGetInt(NULL, NULL, "-n", &nelems, NULL);
    PetscOptionsGetInt(NULL, NULL, "-k", &order, NULL);
    PetscOptionsGetString(NULL, NULL, "-D", dtype, 1024, NULL);

    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Load system from file
    PetscPrintf(PETSC_COMM_WORLD, "Loading initial system\n");
    loadMatsFromFile(&K, &M, dim, dtype, order, nelems);

    // Compute eigenbasis
    PetscPrintf(PETSC_COMM_WORLD, "Initializing eigenbasis solver\n");
    solver.init(K, M, &options);
    PetscPrintf(PETSC_COMM_WORLD, "Computing eigenbasis\n");
    solver.solve(&V, &lambda);
    //solver.solve_simple(K, M, &V, &lambda, &options);

    // PetscInt N, neval;
    // MatGetSize(V, &N, &neval);

    // int size;
    // MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // Run compression experiments
    //double compress1[neval], decompress1[neval], reduce1[neval];
    //double compress2[size],  decompress2[size],  reduce2[size];
    //checkCompress1(V, compress1, decompress1, reduce1, options);
    //checkCompress2(V, compress2, decompress2, reduce2, options);

    // Check accuracy of solution
    //PetscReal norms[neval];
    //checkCorrectness(K, M, V, lambda, norms, options);
    //checkOrthogonality(M, V, options);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double total_elapsed = end_time - start_time;

    if (options.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf\n", total_elapsed);
    else
        PetscPrintf(PETSC_COMM_WORLD, "Total Elapsed: %lf\n", total_elapsed);
    

    // if (options.details())
    // {
    //     // Report
    //     // compress1
    //     for (int i = 0; i < neval; i++)
    //         PetscPrintf(PETSC_COMM_WORLD, "%.12lf ", compress1[i]);
    //     PetscPrintf(PETSC_COMM_WORLD, "\n");

    //     // reduce1
    //     for (int i = 0; i < neval; i++)
    //         PetscPrintf(PETSC_COMM_WORLD, "%.12lf ", reduce1[i]);
    //     PetscPrintf(PETSC_COMM_WORLD, "\n");

    //     // compress2
    //     for (int i = 0; i < size; i++)
    //         PetscPrintf(PETSC_COMM_WORLD, "%.12lf ", compress2[i]);
    //     PetscPrintf(PETSC_COMM_WORLD, "\n");

    //     // reduce2
    //     for (int i = 0; i < size; i++)
    //         PetscPrintf(PETSC_COMM_WORLD, "%.12lf ", reduce2[i]);
    //     PetscPrintf(PETSC_COMM_WORLD, "\n");

    //     // correctness
    //     // for (int i = 0; i < neval; i++)
    //     //     PetscPrintf(PETSC_COMM_WORLD, "%.12lf ", (double) norms[i]);
    //     // PetscPrintf(PETSC_COMM_WORLD, "\n");
    // }

    // Cleanup
    MatDestroy(&K);
    MatDestroy(&M);
    //MatDestroy(&V);
    //VecDestroy(&lambda);

    SlepcFinalize();

    return 0;
}

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems)
{
    PetscViewer viewer;
    char filename[512];

    sprintf(filename, "/scratch/kingspeak/serial/u0450449/fractional/matrices/%dD/%s/%d/stiffness_%d", dim, dtype, order, nelems);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, K);
    MatLoad((*K), viewer);
    MatSetOption((*K), MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin((*K), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*K), MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);

    sprintf(filename, "/scratch/kingspeak/serial/u0450449/fractional/matrices/%dD/%s/%d/mass_%d", dim, dtype, order, nelems);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, M);
    MatLoad((*M), viewer);
    MatSetOption((*M), MAT_HERMITIAN, PETSC_TRUE);
    MatAssemblyBegin((*M), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd((*M), MAT_FINAL_ASSEMBLY);
    PetscViewerDestroy(&viewer);
}

void checkCorrectness(Mat K, Mat M, Mat V, Vec lambda, PetscReal *norms, SolverOptions options)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // residual = K*V - M*V*L;

    // 1) temp = V
    // 2) temp = residual*L
    // 3) residual = M*temp
    // 4) temp = K*V
    // 5) residual = -1 * residual + temp (Y = a*Y + X) (AYPX)

    // residual = -1 * (M*V*L) + (K*V) = KV - MVL 

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    Mat residual, temp;
    MatConvert(V, MATSAME, MAT_INITIAL_MATRIX, &temp);
    MatDiagonalScale(temp, NULL, lambda);
    MatMatMult(M, temp, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &residual);
    MatMatMult(K, V, MAT_REUSE_MATRIX, PETSC_DEFAULT, &temp);
    MatAYPX(residual, -1, temp, DIFFERENT_NONZERO_PATTERN);

    //PetscReal norms[neval];
    MatGetColumnNorms(residual, NORM_2, norms);

    PetscReal minnorm = MPIU_MAX;
    PetscReal maxnorm = 0.0;
    PetscReal avgnorm = 0.0;
    for (int k = 0; k < neval; k++)
    {
        minnorm = (minnorm < norms[k]) ? minnorm : norms[k];
        maxnorm = (maxnorm > norms[k]) ? maxnorm : norms[k];
        avgnorm += norms[k];
    }
    avgnorm /= neval;

    MatDestroy(&residual);
    MatDestroy(&temp);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (options.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf %lf %lf\n", elapsed, start_time, end_time, (double) minnorm, (double) maxnorm, (double) avgnorm);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(checkCorrectness) ||A*vk - lambdak*M*vk|| (min/max/avg) = (%.16lf / %.16lf / %.16lf), Elapsed = %lf, Start = %lf, End = %lf\n", (double) minnorm, (double) maxnorm, (double) avgnorm, elapsed, start_time, end_time);

    //*out_norms = &norms[0];
}

void checkOrthogonality(Mat M, Mat V, SolverOptions options)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    // residual = eye(neval) - V' * M * V

    // temp = M*V
    // residual = V'*temp
    // residual = residual - diag(ones)

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

    if (options.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf", elapsed, start_time, end_time, (double) norm);
    else
        PetscPrintf(PETSC_COMM_WORLD, "(Orthogonality Check) ||I - V'*V|| = %lf, Elapsed = %lf, Start = %lf, End = %lf\n", (double) norm, elapsed, start_time, end_time);
}

void checkCompress1(Mat V, double *compress_times, double *decompress_times, double *data_reduction, SolverOptions options)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // per-process: compress whole eigenvectors independently

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    //double compress_times[neval];
    //double decompress_times[neval];
    //double data_reduction[neval];

    // Single process:
    // For each k:
    //   1) Get local copy of column k of V
    //   2) Time: Compress local array
    //   3) Compute data reduction
    //   4) Time: Decompress local array
    //   5) Report condensed results
    //   7) Write full results to file

    Vec vk_world;
    PetscReal *vk_local;
    PetscInt localsize;

    std::vector<PetscInt> localsizes(size);
    std::vector<PetscInt> displs(size);
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &vk_world);
    MatGetColumnVector(V, vk_world, 0);
    VecGetLocalSize(vk_world, &localsize);
    localsizes[rank] = localsize;
    for (int p = 0; p < size; p++)
        MPI_Bcast(&localsizes[p], 1, MPIU_INT, p, PETSC_COMM_WORLD);
    displs[0] = 0;
    for (int p = 1; p < size; p++)
        displs[p] = displs[p-1] + localsizes[p-1];
    
    std::vector<PetscReal> vk;
    vk.resize(N);
    for (int k = 0; k < neval/size; k++)
    {
        for (int i = 0; i < size; i++)
        {
            MatGetColumnVector(V, vk_world, size*k + i);
            VecGetArray(vk_world, &vk_local);
            MPI_Gatherv(vk_local, localsize, MPIU_REAL, &vk[0], 
                &localsizes[0], &displs[0], MPIU_REAL, i, 
                PETSC_COMM_WORLD);
        }

        // compress vk
        double compress_start_time = MPI_Wtime();
        double reduction = compress1D(&vk[0], N, 1e-14, 0);
        double compress_stop_time = MPI_Wtime();
        compress_times[size*k+rank] = compress_stop_time - compress_start_time;
        data_reduction[size*k+rank] = reduction;

        for (int i = 0; i < size; i++)
        {
            MPI_Bcast(&compress_times[size*k + i], 1, MPI_DOUBLE, i, PETSC_COMM_WORLD);
        }

        VecRestoreArray(vk_world, &vk_local);
    }

    // report:
    //   min/max/avg compress time
    //   min/max/avg decompress time
    //   min/max/avg data reduction

    double min_compress = MPIU_MAX, max_compress = 0.0, avg_compress = 0.0;
    //double min_decompress = MPIU_MAX, max_decompress = 0.0, avg_decompress = 0.0;
    double min_reduce = MPIU_MAX, max_reduce = 0.0, avg_reduce = 0.0;
    if (rank == 0)
    {
        for (int k = 0; k < neval; k++)
        {
            min_compress   = (min_compress < compress_times[k])     ? min_compress   : compress_times[k];
            //min_decompress = (min_decompress < decompress_times[k]) ? min_decompress : decompress_times[k];
            min_reduce     = (min_reduce < data_reduction[k])       ? min_reduce     : data_reduction[k];

            max_compress   = (max_compress > compress_times[k])     ? max_compress   : compress_times[k];
            //max_decompress = (max_decompress > decompress_times[k]) ? max_decompress : decompress_times[k];
            max_reduce     = (max_reduce > data_reduction[k])       ? max_reduce     : data_reduction[k];

            avg_compress   += compress_times[k];
            //avg_decompress += decompress_times[k];
            avg_reduce     += data_reduction[k];
        }

        avg_compress   /= neval;
        //avg_decompress /= neval;
        avg_reduce     /= neval;

        printf("Compression time   (min/max/avg): %lf, %lf, %lf\n", min_compress,   max_compress,   avg_compress);
        //printf("Decompression time (min/max/avg): %lf, %lf, %lf\n", min_decompress, max_decompress, avg_decompress);
        printf("Data reduction     (min/max/avg): %lf, %lf, %lf\n", min_reduce,     max_reduce,     avg_reduce);
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    VecDestroy(&vk_world);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (options.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", elapsed, start_time, end_time, min_compress, max_compress, avg_compress, min_reduce, max_reduce, avg_reduce);
    else
    {
        PetscPrintf(PETSC_COMM_WORLD, "(Compression Experiment 1)\n");
        PetscPrintf(PETSC_COMM_WORLD, "  Elapsed = %lf, Start = %lf, End = %lf\n", elapsed, start_time, end_time);
        PetscPrintf(PETSC_COMM_WORLD, "  Time to Compress: (min/max/avg) = (%lf / %lf / %lf)\n", min_compress, max_compress, avg_compress);
        PetscPrintf(PETSC_COMM_WORLD, "  Data Reduction:   (min/max/avg) = (%lf / %lf / %lf)\n", min_reduce, max_reduce, avg_reduce);
    }

    //*out_compress = &compress_times[0];
    //*out_decompress = decompress_times;
    //*out_reduce = &data_reduction[0];
}
void checkCompress2(Mat V, double *compress_times, double *decompress_times, double *data_reduction, SolverOptions options)
{
    MPI_Barrier(PETSC_COMM_WORLD);
    double start_time = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // per-process: compress local data block

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    //double compress_times[size];
    //double decompress_times[size];
    //double data_reduction[size];

    // Each process p:
    // 1) Get local block from V
    // 2) Time: Compress local block
    // 3) Compute data reduction
    // 4) Time: Decompress local block
    // 5) Report condensed results
    // 6) Write full results to file

    PetscInt m,n;
    MatGetLocalSize(V, &m, &n);

    PetscReal *v_block;
    MatDenseGetArray(V, &v_block);

    double compress_start_time = MPI_Wtime();
    double reduction = compress2D(&v_block[0], m, n, 1e-14, 0);
    double compress_stop_time = MPI_Wtime();
    compress_times[rank] = compress_stop_time - compress_start_time;
    data_reduction[rank] = reduction;

    for (int p = 0; p < size; p++)
    {
        MPI_Bcast(&compress_times[p],   1, MPI_DOUBLE, p, PETSC_COMM_WORLD);
        MPI_Bcast(&decompress_times[p], 1, MPI_DOUBLE, p, PETSC_COMM_WORLD);
        MPI_Bcast(&data_reduction[p],   1, MPI_DOUBLE, p, PETSC_COMM_WORLD);
    }

    // report:
    //   min/max/avg compress time
    //   min/max/avg decompress time
    //   min/max/avg data reduction

    double min_compress = MPIU_MAX, max_compress = 0.0, avg_compress = 0.0;
    //double min_decompress = MPIU_MAX, max_decompress = 0.0, avg_decompress = 0.0;
    double min_reduce = MPIU_MAX, max_reduce = 0.0, avg_reduce = 0.0;
    for (int p = 0; p < size; p++)
    {
        min_compress   = (min_compress < compress_times[p])     ? min_compress   : compress_times[p];
        //min_decompress = (min_decompress < decompress_times[p]) ? min_decompress : decompress_times[p];
        min_reduce     = (min_reduce < data_reduction[p])       ? min_reduce     : data_reduction[p];

        max_compress   = (max_compress > compress_times[p])     ? max_compress   : compress_times[p];
        //max_decompress = (max_decompress > decompress_times[p]) ? max_decompress : decompress_times[p];
        max_reduce     = (max_reduce > data_reduction[p])       ? max_reduce     : data_reduction[p];

        avg_compress   += compress_times[p];
        //avg_decompress += decompress_times[p];
        avg_reduce     += data_reduction[p];
    }
    avg_compress   /= size;
    //avg_decompress /= size;
    avg_reduce     /= size;
    PetscPrintf(PETSC_COMM_WORLD, "Compression time   (min/max/avg): %lf, %lf, %lf\n", min_compress,   max_compress,   avg_compress);
    //PetscPrintf(PETSC_COMM_WORLD, "Decompression time (min/max/avg): %lf, %lf, %lf\n", min_decompress, max_decompress, avg_decompress);
    PetscPrintf(PETSC_COMM_WORLD, "Data Reduction     (min/max/avg): %lf, %lf, %lf\n", min_reduce,     max_reduce,     avg_reduce);

    MPI_Barrier(PETSC_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    if (options.terse())
        PetscPrintf(PETSC_COMM_WORLD, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", elapsed, start_time, end_time, min_compress, max_compress, avg_compress, min_reduce, max_reduce, avg_reduce);
    else
    {
        PetscPrintf(PETSC_COMM_WORLD, "(Compression Experiment 2)\n");
        PetscPrintf(PETSC_COMM_WORLD, "  Elapsed = %lf, Start = %lf, End = %lf\n", elapsed, start_time, end_time);
        PetscPrintf(PETSC_COMM_WORLD, "  Time to Compress: (min/max/avg) = (%lf / %lf / %lf)\n", min_compress, max_compress, avg_compress);
        PetscPrintf(PETSC_COMM_WORLD, "  Data Reduction:   (min/max/avg) = (%lf / %lf / %lf)\n", min_reduce, max_reduce, avg_reduce);
    }
    
    //*out_compress = &compress_times[0];
    //*out_decompress = decompress_times;
    //*out_reduce = &data_reduction[0];
}

/* compress or decompress array */
double compress1D(double* array, int nx, double tolerance, int decompress)
{
    int status = 0;    /* return value: 0 = success */
    zfp_type type;     /* array scalar type */
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    void* buffer;      /* storage for compressed stream */
    size_t bufsize;    /* byte size of compressed buffer */
    bitstream* stream; /* bit stream to write to or read from */
    size_t zfpsize;    /* byte size of compressed stream */

    /* allocate meta data for the 3D array a[nz][ny][nx] */
    type = zfp_type_double;
    field = zfp_field_1d(array, type, nx);

    /* allocate meta data for a compressed stream */
    zfp = zfp_stream_open(NULL);

    /* set compression mode and parameters via one of three functions */
    /*  zfp_stream_set_rate(zfp, rate, type, 3, 0); */
    /*  zfp_stream_set_precision(zfp, precision); */
    zfp_stream_set_accuracy(zfp, tolerance);

    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    buffer = malloc(bufsize);

    /* associate bit stream with allocated buffer */
    stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    /* compress or decompress entire array */
    if (decompress) 
    {
        /* read compressed stream and decompress array */
        zfpsize = fread(buffer, 1, bufsize, stdin);
        if (!zfp_decompress(zfp, field)) 
        {
            fprintf(stderr, "decompression failed\n");
            status = 1;
        }
    }
    else {
        /* compress array and output compressed stream */
        zfpsize = zfp_compress(zfp, field);
        if (!zfpsize) 
        {
            fprintf(stderr, "compression failed\n");
            status = 1;
        }
        //else
            //fwrite(buffer, 1, zfpsize, stdout);
    }

    double originalsize = nx*sizeof(double);
    double compressedsize = zfpsize;

    //printf("Original size: %lu, Buffer size: %lu, Zfp size: %lu\n", (size_t) nx*sizeof(double), bufsize, zfpsize);

    /* clean up */
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);

    return originalsize / compressedsize;
}

/* compress or decompress array */
double compress2D(double* array, int nx, int ny, double tolerance, int decompress)
{
    int status = 0;    /* return value: 0 = success */
    zfp_type type;     /* array scalar type */
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    void* buffer;      /* storage for compressed stream */
    size_t bufsize;    /* byte size of compressed buffer */
    bitstream* stream; /* bit stream to write to or read from */
    size_t zfpsize;    /* byte size of compressed stream */

    /* allocate meta data for the 3D array a[nz][ny][nx] */
    type = zfp_type_double;
    field = zfp_field_2d(array, type, nx, ny);

    /* allocate meta data for a compressed stream */
    zfp = zfp_stream_open(NULL);

    /* set compression mode and parameters via one of three functions */
    /*  zfp_stream_set_rate(zfp, rate, type, 3, 0); */
    /*  zfp_stream_set_precision(zfp, precision); */
    zfp_stream_set_accuracy(zfp, tolerance);

    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    buffer = malloc(bufsize);

    /* associate bit stream with allocated buffer */
    stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    /* compress or decompress entire array */
    if (decompress) 
    {
        /* read compressed stream and decompress array */
        zfpsize = fread(buffer, 1, bufsize, stdin);
        if (!zfp_decompress(zfp, field)) 
        {
            fprintf(stderr, "decompression failed\n");
            status = 1;
        }
    }
    else {
        /* compress array and output compressed stream */
        zfpsize = zfp_compress(zfp, field);
        if (!zfpsize) 
        {
            fprintf(stderr, "compression failed\n");
            status = 1;
        }
        //else
            //fwrite(buffer, 1, zfpsize, stdout);
    }

    double originalsize = nx*ny*sizeof(double);
    double compressedsize = zfpsize;

    /* clean up */
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);

    return originalsize / compressedsize;
}