#include "eigen_mm.h"

void loadMatsFromFile(Mat *K, Mat *M, 
    PetscInt dim, const char *dtype, PetscInt order, 
    PetscInt nelems);
void checkCorrectness(Mat K, Mat M, Mat V, Vec lambda);
void checkOrthogonality(Mat M, Mat V);
void checkCompress1(Mat V);
void checkCompress2(Mat V);
double compress1D(double* array, int nx, double tolerance, int decompress);
double compress2D(double* array, int nx, int ny, double tolerance, int decompress);

int main(int argc, char *argv[])
{
    Mat K, M, V;
    Vec lambda;
    SolverOptions options;
    eigen_mm solver;

    options.nevt = 100;
    options.splitmaxiters = 10;
    options.nodesperevaluator = 1;
    options.subproblemsperevaluator = 16;
    options.taskspernode = 28;
    options.splittol = 1e-3;
    options.p = 30;
    options.nv = 10;
    options.raditers = 10;
    options.radtol = 1e-3;
    options.L = 0.01;
    options.R = 10000.0;

    SlepcInitialize(NULL,NULL,NULL,NULL);
    PetscPrintf(PETSC_COMM_WORLD, "Slepc has been initialized\n");

    PetscPrintf(PETSC_COMM_WORLD, "Loading global input matrix\n");
    loadMatsFromFile(&K, &M, 2, "square", 1, 49);

    PetscPrintf(PETSC_COMM_WORLD, "Initializing solver\n");
    solver.init(K, M, options);
    PetscPrintf(PETSC_COMM_WORLD, "Running solver\n");
    solver.solve(&V, &lambda);

    // Run compression experiment
    checkCompress1(V);
    checkCompress2(V);

    // Check accuracy of solution
    checkCorrectness(K,M,V,lambda);
    checkOrthogonality(M,V);

    // Compression experiments

    PetscPrintf(PETSC_COMM_WORLD, "Finalizing SLEPC\n");
    MatDestroy(&K);
    MatDestroy(&M);
    MatDestroy(&V);
    VecDestroy(&lambda);
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

void checkCorrectness(Mat K, Mat M, Mat V, Vec lambda)
{
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

    PetscReal norms[neval];
    MatGetColumnNorms(residual, NORM_2, norms);

    PetscReal minnorm = MPIU_MAX;
    PetscReal maxnorm = 0.0;
    for (int k = 0; k < neval; k++)
    {
        minnorm = (minnorm < norms[k]) ? minnorm : norms[k];
        maxnorm = (maxnorm > norms[k]) ? maxnorm : norms[k];
    }

    PetscPrintf(PETSC_COMM_WORLD, "Minimum eigenpair error: %.16lf\n", (double) minnorm);
    PetscPrintf(PETSC_COMM_WORLD, "Maximum eigenpair error: %.16lf\n", (double) maxnorm);

    MatDestroy(&residual);
    MatDestroy(&temp);
}

void checkOrthogonality(Mat M, Mat V)
{
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

    PetscPrintf(PETSC_COMM_WORLD, "Orthogonality Norm: %.16lf\n", norm);

    VecDestroy(&ones);
    MatDestroy(&residual);
    MatDestroy(&temp);
}

void checkCompress1(Mat V)
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // per-process: compress whole eigenvectors independently

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    double compress_times[neval];
    double decompress_times[neval];
    double data_reduction[neval];

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
    if (rank == 0) vk.resize(N);
    for (int k = 0; k < neval; k++)
    {
        MatGetColumnVector(V, vk_world, k);
        VecGetArray(vk_world, &vk_local);
        MPI_Gatherv(vk_local, localsize, MPIU_REAL, &vk[0], 
            &localsizes[0], &displs[0], MPIU_REAL, 0, 
            PETSC_COMM_WORLD);

        if (rank == 0)
        {
            // compress vk
            double start_time = MPI_Wtime();
            double reduction = compress1D(&vk[0], N, 1e-14, 0);
            double stop_time = MPI_Wtime();
            compress_times[k] = stop_time - start_time;
            data_reduction[k] = reduction;
        }
        VecRestoreArray(vk_world, &vk_local);
        MPI_Barrier(PETSC_COMM_WORLD);
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

    // write to file full statistics
}
void checkCompress2(Mat V)
{
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // per-process: compress local data block

    PetscInt N, neval;
    MatGetSize(V, &N, &neval);

    double compress_times[size];
    double decompress_times[size];
    double data_reduction[size];

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

    double start_time = MPI_Wtime();
    double reduction = compress2D(&v_block[0], m, n, 1e-14, 0);
    double stop_time = MPI_Wtime();
    compress_times[rank] = stop_time - start_time;
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

    // write to file full statistics
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