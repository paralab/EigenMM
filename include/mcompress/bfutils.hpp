// utilities
double frob_norm_block(Mat &A, BFInt idx)
{
    double retval = 0.0;
    for (BFInt row = idx; row < A.getRows(); row++)
    {
        for (BFInt col = row; col < A.getCols(); col++)
        {
            retval += A(row, col)*A(row,col);
        }
    }
    return sqrt(retval);
}

double compare_mats(Mat &A, Mat &B)
{
    double upper = 0.0;
    double lower = 0.0;
    for (BFInt row = 0; row < A.getRows(); row++)
    {
        for (BFInt col = 0; col < A.getCols(); col++)
        {
            upper += pow((A(row, col) - B(row, col)), 2.0);
            lower += pow(A(row, col), 2.0);
        }
    }
    return sqrt(upper)/sqrt(lower);
}
void bf_init(BFInt L, std::vector<std::vector<BFInt>> &ids)
{
    BFInt len, nsegs, id0;
    BFInt P = (1 << L);
    ids.resize(L);
    for (BFInt l = 0; l < L; l++)
    {
        ids[l].resize(2*P);
        len = (2*P) >> l;
        nsegs = (2*P) / len;
        id0 = 0;
        for(BFInt s = 0; s < nsegs; s++)
        {
            for (BFInt i = 0; i < len/2; i++)
                ids[l][id0 + i] = id0 + i*2;
            for (BFInt i = len/2; i < len; i++)
                ids[l][id0 + i] = id0 + (i-len/2)*2 + 1;
            id0 += len;
        }
    }
}

PetscInt reverse_int(PetscInt v)
{
    PetscInt o;
    char *src = (char *) &v;
    char *dst = (char *) &o;
    
    for (PetscInt i = 0; i < sizeof(PetscInt); i++)
        dst[i] = src[sizeof(PetscInt)-1-i];

    return o;
}
double reverse_double(double v)
{
    double o;
    char *src = (char *) &v;
    char *dst = (char *) &o;
    
    for (BFInt i = 0; i < sizeof(double); i++)
        dst[i] = src[sizeof(double)-1-i];

    return o;
}

void petsc_binary_read(std::string filename, std::vector<double> &Adata, 
    BFInt *nRows_out, BFInt *nCols_out)
{
    // PetscInt    MAT_FILE_CLASSID
    // PetscInt    number of rows
    // PetscInt    number of columns
    // PetscInt    total number of nonzeros
    // PetscInt    *number nonzeros in each row
    // PetscInt    *column indices of all nonzeros (starting index is zero)
    // PetscScalar *values of all nonzeros

    std::ifstream in_stream;
    in_stream.open(filename, std::ios::binary);

    // get MAT_FILE_CLASSID
    PetscInt classid;
    in_stream.read((char *) &classid, sizeof(PetscInt));
    classid = reverse_int(classid);

    // get nRows, nCols, and nNz
    PetscInt nRows, nCols, nNz;
    in_stream.read((char *) &nRows, sizeof(PetscInt));
    in_stream.read((char *) &nCols, sizeof(PetscInt));
    in_stream.read((char *) &nNz,   sizeof(PetscInt));
    nRows = reverse_int(nRows);
    nCols = reverse_int(nCols);
    nNz   = nRows*nCols;
    // printf("classid = %d, nRows = %d, nCols = %d, nNz = %d\n", classid, nRows, nCols, nNz);

    BFInt long_nnz = ((BFInt) nRows) * ((BFInt) nCols);

    Adata.resize(long_nnz);
    in_stream.read((char *) &Adata[0], long_nnz*sizeof(double));
    for (BFInt i = 0; i < long_nnz; i++)
        Adata[i] = reverse_double(Adata[i]);

    nRows_out[0] = (BFInt) nRows;
    nCols_out[0] = (BFInt) nCols;

    in_stream.close();
    // printf("petsc binary read complete\n");
}

void load_data_from_file(std::string filename, std::vector<double> &Adata, 
    BFInt *nRows, BFInt *nCols)
{
    std::ifstream in_stream;
    in_stream.open(filename, std::ios::binary);

    in_stream.read((char *) nRows, sizeof(BFInt));
    in_stream.read((char *) nCols, sizeof(BFInt));

    Adata.resize(nRows[0] * nCols[0]);
    in_stream.read((char *) &Adata[0], nRows[0]*nCols[0]*sizeof(double)); 

    in_stream.close();
}

void generate_test_data(std::string filename)
{
    BFInt nRows = 1000;
    BFInt nCols = 1000;
    std::vector<BFInt> matrixinfo;
    matrixinfo.push_back(nRows);
    matrixinfo.push_back(nCols);

    std::vector<double> matrixdata;
    for (BFInt row = 0; row < nRows; row++)
        for (BFInt col = 0; col < nCols; col++)
            matrixdata.push_back(col + row*nCols);

    // std::random_device rng_device;
    // std::default_random_engine rng_engine(rng_device());
    // std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
    // for (int row = 0; row < nRows; row++)
    //     for (int col = 0; col < nCols; col++)
    //         matrixdata.push_back(uniform_dist(rng_engine));

    std::ofstream out_stream(filename, std::ios::out | std::ios::binary);
    out_stream.write((char*)&matrixinfo[0], matrixinfo.size() * sizeof(BFInt));
    out_stream.write((char*)&matrixdata[0], matrixdata.size() * sizeof(double));
    out_stream.close();
}

void bf_save(std::string filename, BFMat &bfA)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<double> buffer;
    std::vector<BFInt> pivbuffer;
    std::vector<BFInt> pivinvbuffer;

    std::ofstream out_stream;

    BFInt globalRows = bfA.getGlobalRows();
    BFInt globalCols = bfA.getGlobalCols();
    BFInt L = bfA.getL();
    BFInt P = 1 << L;
    double compressionRatio = bfA.getCompressionRatio();
    double inputTolerance = bfA.getInputTolerance();

    std::vector<MPI_Status> send_status(6*P);
    std::vector<MPI_Status> recv_status(6*P);

    if (rank == 0)
    {
        out_stream.open(filename, std::ios::out | std::ios::binary);
        out_stream.write((char*) &globalRows, sizeof(BFInt));
        out_stream.write((char*) &globalCols, sizeof(BFInt));
        out_stream.write((char*) &L, sizeof(BFInt));
        out_stream.write((char*) &compressionRatio, sizeof(double));
        out_stream.write((char*) &inputTolerance, sizeof(double));
        // printf("Writing bfA file (%s) with %ld rows, %ld cols, L = %ld, cRatio = %lf, tol = %lf, sizeof(BFInt) = %lu\n", 
        //     filename.c_str(), globalRows, globalCols, L, compressionRatio, 
        //     inputTolerance, sizeof(BFInt));
        out_stream.write((char*) bfA.getAaRowData(), 2*P*sizeof(BFInt));
        out_stream.write((char*) bfA.getAaColData(), 2*P*sizeof(BFInt));
        for (BFInt l = 0; l <= L; l++)
        {
            out_stream.write((char*) bfA.getAbRowData(l), 2*P*sizeof(BFInt));
            out_stream.write((char*) bfA.getAbColData(l), 2*P*sizeof(BFInt));
        }        
    }

    for (BFInt j = 0; j < 2*P; j++)
    {
        BFInt m = bfA.getAaRows(j);
        BFInt n = bfA.getAaCols(j);
        if (j % size != 0)
        {
            if (rank == j % size) MPI_Send(bfA.getAaData(j), m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            if (rank == 0)
            {
                buffer.resize(m*n);
                MPI_Recv(&buffer[0], m*n, MPI_DOUBLE, j % size, 0, MPI_COMM_WORLD, &recv_status[3*j]);
                out_stream.write((char*) &buffer[0], m*n*sizeof(double));
            }
        }
        else
        {
            if (rank == 0) out_stream.write((char*) bfA.getAaData(j), m*n*sizeof(double));
        }
    }

    for (BFInt l = 0; l <= L; l++)
    {
        for (BFInt j = 0; j < 2*P; j++)
        {
            BFInt m = bfA.getAbRows(l,j);
            BFInt n = bfA.getAbCols(l,j);
            if (j % size != 0)
            {
                if (rank == j % size) 
                {
                    MPI_Send(bfA.getAbTData(l,j), m*(n-m), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(bfA.getAbPivData(l,j), n, MPI_BFINT, 0, 0, MPI_COMM_WORLD);
                    MPI_Send(bfA.getAbPivInvData(l,j), n, MPI_BFINT, 0, 0, MPI_COMM_WORLD);
                }
                if (rank == 0) 
                {
                    buffer.resize(m*(n-m));
                    MPI_Recv(&buffer[0], m*(n-m), MPI_DOUBLE, j % size, 0, MPI_COMM_WORLD, &recv_status[3*j]);
                    out_stream.write((char*) &buffer[0], m*(n-m)*sizeof(double));

                    pivbuffer.resize(n);
                    MPI_Recv(&pivbuffer[0], n, MPI_BFINT, j % size, 0, MPI_COMM_WORLD, &recv_status[3*j+1]);
                    out_stream.write((char*) &pivbuffer[0], n*sizeof(BFInt));

                    pivinvbuffer.resize(n);
                    MPI_Recv(&pivinvbuffer[0], n, MPI_BFINT, j % size, 0, MPI_COMM_WORLD, &recv_status[3*j+2]);
                    out_stream.write((char*) &pivinvbuffer[0], n*sizeof(BFInt));
                }
            }
            else
            {
                if (rank == 0) 
                {
                    out_stream.write((char*) bfA.getAbTData(l,j), m*(n-m)*sizeof(double));
                    out_stream.write((char*) bfA.getAbPivData(l,j), n*sizeof(BFInt));
                    out_stream.write((char*) bfA.getAbPivInvData(l,j), n*sizeof(BFInt));
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) out_stream.close();
}
void compute_loads(BFMat &bfA, std::vector<std::vector<BFInt>> &loads)
{
    BFInt L = bfA.getL();
    BFInt P = 1 << L;
    for (BFInt l = 0; l <= L; l++)
    {
        for (BFInt j = 0; j < 2*P; j++)
        {
            loads[l][j] = bfA.getAbRows(l,j) * (bfA.getAbCols(l,j) - bfA.getAbRows(l,j));
            if (l == L) loads[l][j] += bfA.getAaRows(j) * bfA.getAaCols(j);
        }
    }
}
void pair_balance(BFInt L, BFInt size, std::vector<std::vector<BFInt>> &loads, std::vector<std::vector<BFInt>> &owned_ids)
{
    BFInt P = 1 << L;
    for (BFInt l = 0; l <= L; l++)
    {
        std::vector<BFInt> idx(loads[l].size());
        std::vector<BFInt> loads_l(loads[l]);
        for (BFInt j = 0; j < idx.size(); j++)
            idx[j] = j;
        std::stable_sort(idx.begin(), idx.end(), 
            [&loads_l](BFInt i1, BFInt i2) { return loads_l[i1] < loads_l[i2]; } );
        for (BFInt j = 0; j < P; j++)
        {
            owned_ids[l][idx[j]]           = j % size;
            owned_ids[l][idx[2*P - j - 1]] = j % size;
        }
    }
}
void bf_load(std::string filename, BFMat &bfA)
{
    //   determine which process gets which matrices (naive approach)
    //     distribute matrices in a round robin approach

    MPI_Status status;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BFInt globalRows, globalCols;
    BFInt L, P;
    double compressionRatio;
    double inputTolerance;
    std::vector<BFInt> AaRows;
    std::vector<BFInt> AaCols;
    std::vector<std::vector<BFInt>> AbRows;
    std::vector<std::vector<BFInt>> AbCols;
    std::vector<double> databuffer;
    std::vector<BFInt> pivbuffer;
    std::vector<BFInt> pivinvbuffer;
    std::vector<std::vector<BFInt>> owned_ids;
    std::vector<std::vector<BFInt>> loads;
    std::ifstream in_stream;
    in_stream.open(filename, std::ios::binary);

    if (rank == 0)
    {
        in_stream.read((char *) &globalRows, sizeof(BFInt));
        in_stream.read((char *) &globalCols, sizeof(BFInt));
        in_stream.read((char *) &L, sizeof(BFInt));
        in_stream.read((char *) &compressionRatio, sizeof(double));
        in_stream.read((char *) &inputTolerance, sizeof(double));
    }
    MPI_Bcast(&globalRows, 1, MPI_BFINT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&globalCols, 1, MPI_BFINT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L, 1, MPI_BFINT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&compressionRatio, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&inputTolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // printf("Reading bfA file (%s) with %ld rows, %ld cols, L = %ld, cRatio = %lf, tol = %lf\n", 
    //     filename.c_str(), globalRows, globalCols, L, compressionRatio, 
    //     inputTolerance);
    
    P = 1 << L;

    bfA.init(L);
    bfA.setGlobalRows(globalRows);
    bfA.setGlobalCols(globalCols);
    bfA.setCompressionRatio(compressionRatio);
    bfA.setInputTolerance(inputTolerance);

    loads.resize(L+1);
    owned_ids.resize(L+1);
    for (BFInt l = 0; l <= L; l++)
    {
        loads[l].resize(2*P);
        owned_ids[l].resize(2*P);
    }

    if (rank == 0)
    {
        in_stream.read((char *) bfA.getAaRowData(), 2*P*sizeof(BFInt));
        in_stream.read((char *) bfA.getAaColData(), 2*P*sizeof(BFInt));
        for (BFInt l = 0; l <= L; l++)
        {
            in_stream.read((char *) bfA.getAbRowData(l), 2*P*sizeof(BFInt));
            in_stream.read((char *) bfA.getAbColData(l), 2*P*sizeof(BFInt));
        }

        compute_loads(bfA, loads);
        pair_balance(L, size, loads, owned_ids);
    }
    MPI_Bcast(bfA.getAaRowData(), 2*P, MPI_BFINT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bfA.getAaColData(), 2*P, MPI_BFINT, 0, MPI_COMM_WORLD);
    for (BFInt l = 0; l <= L; l++)
    {
        MPI_Bcast(&owned_ids[l][0], 2*P, MPI_BFINT, 0, MPI_COMM_WORLD);
        MPI_Bcast(bfA.getAbRowData(l), 2*P, MPI_BFINT, 0, MPI_COMM_WORLD);
        MPI_Bcast(bfA.getAbColData(l), 2*P, MPI_BFINT, 0, MPI_COMM_WORLD);
    }
    // for (BFInt l = 0; l <= L; l++)
    // {
    //     for (BFInt j = 0; j < 2*P; j++)
    //     {
    //         printf("(%d x %d) ", bfA.getAbRows(l,j), bfA.getAbCols(l,j));
    //     }
    //     printf("\n");
    // }
    bfA.setOwnedMap(owned_ids);

    // Aa
    for (BFInt j = 0; j < 2*P; j++)
    {
        BFInt m = bfA.getAaRows(j);
        BFInt n = bfA.getAaCols(j);
        if (rank == 0)
        {
            databuffer.resize(m*n);
            in_stream.read((char *) &databuffer[0], m*n*sizeof(double));
        }
        if (owned_ids[L][j] == 0)
        {
            // printf("[%d] Setting matrix Aa[%d] (%d by %d)\n", rank, j, m, n);
            if (rank == 0) bfA.setAa(j, m, n, databuffer);
        }
        else
        {
            if (rank == 0)
            {
                MPI_Send(&databuffer[0], m*n, MPI_DOUBLE, owned_ids[L][j], 0, MPI_COMM_WORLD);
            }
            if (rank == owned_ids[L][j])
            {
                databuffer.resize(m*n);
                MPI_Recv(&databuffer[0], m*n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
                // printf("[%d] Setting matrix Aa[%d] (%d by %d)\n", rank, j, m, n);
                bfA.setAa(j, m, n, databuffer);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Ab
    for (BFInt l = 0; l <= L; l++)
    {
        for (BFInt j = 0; j < 2*P; j++)
        {
            BFInt m = bfA.getAbRows(l,j);
            BFInt n = bfA.getAbCols(l,j);
            if (rank == 0)
            {
                databuffer.resize(m*n);
                pivbuffer.resize(n);
                pivinvbuffer.resize(n);
                in_stream.read((char *) &databuffer[0], m*(n-m)*sizeof(double));
                in_stream.read((char *) &pivbuffer[0], n*sizeof(BFInt));
                in_stream.read((char *) &pivinvbuffer[0], n*sizeof(BFInt));
            }
            if (owned_ids[l][j] == 0)
            {
                // printf("[%d] Setting matrix Ab[%d][%d] (%d by %d)\n", rank, l, j, m, n);
                if (rank == 0) bfA.setAb(l, j, m, n, databuffer, pivbuffer, pivinvbuffer);
            }
            else
            {
                if (rank == 0)
                {
                    MPI_Send(&databuffer[0], m*(n-m), MPI_DOUBLE, owned_ids[l][j], 0, MPI_COMM_WORLD);
                    MPI_Send(&pivbuffer[0], n, MPI_BFINT, owned_ids[l][j], 0, MPI_COMM_WORLD);
                    MPI_Send(&pivinvbuffer[0], n, MPI_BFINT, owned_ids[l][j], 0, MPI_COMM_WORLD);
                }
                if (rank == owned_ids[l][j])
                {
                    databuffer.resize(m*n);
                    pivbuffer.resize(n);
                    pivinvbuffer.resize(n);
                    MPI_Recv(&databuffer[0], m*(n-m), MPI_DOUBLE, 0 , 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&pivbuffer[0], n, MPI_BFINT, 0, 0, MPI_COMM_WORLD, &status);
                    MPI_Recv(&pivinvbuffer[0], n, MPI_BFINT, 0, 0, MPI_COMM_WORLD, &status);
                    // printf("[%d] Setting matrix Ab[%d][%d] (%d by %d)\n", rank, l, j, m, n);
                    bfA.setAb(l, j, m, n, databuffer, pivbuffer, pivinvbuffer);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // Each process cycles through their owned matrices to find max_m and max_n
    BFInt max_m = 0;
    BFInt max_n = 0;
    BFInt max_j = 0;
    for (BFInt l = 0; l <= L; l++)
    {
        max_j = std::max(max_j, bfA.getNumberOwned(l));
        for (BFInt j = 0; j < bfA.getNumberOwned(l); j++)
        {
            max_m = std::max(max_m, bfA.getAbRows(l, bfA.getOwned(l,j)));
            max_n = std::max(max_n, bfA.getAbCols(l, bfA.getOwned(l,j)));
            if (l == L) max_n = std::max(max_n, bfA.getAaRows(bfA.getOwned(l,j)));
        }
    }
    for (BFInt j = 0; j < max_j; j++)
    {
        bfA.initializeXp(max_n);
        bfA.initializeYp(max_m);
    }
    bfA.initializeWorkspace();
}