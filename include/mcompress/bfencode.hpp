
void bf_partition(BFMat &bfA, Mat *A_in, std::vector<Mat *> &A)
{
    BFInt L = bfA.getL();
    BFInt P = 1 << L;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BFInt nRows = bfA.getGlobalRows();
    BFInt nCols = bfA.getGlobalCols();

    BFInt lRows, lCols;
    BFInt row0 = 0;
    BFInt col0 = 0;
    int count = 0;
    std::vector<double> lData;
    std::vector<MPI_Request> sendRequests(2*P);
    std::vector<MPI_Status> sendStatuses(2*P);
    std::vector<MPI_Status> recvStatuses(2*P);
    for(BFInt col = 0; col < P; col++)
    {
        row0 = 0;
        lCols = (nCols+col)/P;
        for(BFInt row = 0; row < 2; row++)
        {
            lRows = (nRows+row)/((BFInt)2);
            if (count > 0 && rank == 0) MPI_Wait(&sendRequests[count-1], &sendStatuses[count-1]);
            lData.resize(lRows * lCols);
            if (rank == 0)
            {
                // printf("processing partition (%ld, %ld) out of (%ld, %ld)\n", row, col, P, 2);
                for (BFInt lcol = 0; lcol < lCols; lcol++)
                {
                    for (BFInt lrow = 0; lrow < lRows; lrow++)
                    {
                        lData[lcol + lrow*lCols] = (*A_in)(lrow + row0, lcol + col0);
                    }
                }
                MPI_Isend(&lData[0], lRows*lCols, MPI_DOUBLE, count % size, 0, MPI_COMM_WORLD, &sendRequests[count]);
            }
            if (rank == count % size)
            {
                MPI_Recv(&lData[0], lRows*lCols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &recvStatuses[count]);
                for (BFInt l = 0; l <= L; l++)
                    bfA.setOwned(l, (BFInt) count);
                A[count] = new Mat(lRows, lCols, lData);
            }
            
            count++;
            row0 += lRows;
        }
        col0 += lCols;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

BFInt bf_id_encode(Mat **Aa, IDMat **Ab, Mat &A, double tol)
{
    BFInt m = A.getRows();
    BFInt n = A.getCols();
    BFInt r = 0;
    double nrm = 1.0;

    Mat QR(m,n,A.getData());
    std::vector<int> piv(n);
    std::vector<int> pivinv(n);
    std::vector<BFInt> longpiv(n);
    std::vector<BFInt> longpivinv(n);
    std::vector<double> tau(n);
    for (BFInt i = 0; i < n; i++)
        piv[i] = 0;
    for (BFInt i = 0; i < n; i++)
        pivinv[i] = i;

    // AP = QR
    BFInt info = LAPACKE_dgeqp3(LAPACK_ROW_MAJOR, m, n, QR.data(), n, (int *) &piv[0], &tau[0]);

    // upper diagonal of 'QR' is R
    // lower diagonal of 'QR' and 'tau' are used to apply or construct Q

    // Determine numerical rank
    while (nrm > tol && r < n)
    {
        nrm = frob_norm_block(QR, r);
        r++;
    }
    if (nrm <= tol) r--;
    // if (r == 0) printf("[Invalid Rank Detected] A: (%d by %d), nrm = %lf, tol = %lf, r = %d, info = %d\n", m, n, nrm, tol, r, info);

    // Form R11 and R12
    Mat R11(r, r);
    Mat R12(r, n-r);
    for (BFInt row = 0; row < r; row++)
        for (BFInt col = 0; col < r; col++)
            R11(row, col) = 0.0;
    for (BFInt row = 0; row < r; row++)
        for (BFInt col = row; col < r; col++)
            R11(row, col) = QR(row, col);
    for (BFInt row = 0; row < r; row++)
        for (BFInt col = 0; col < n-r; col++)
            R12(row, col) = QR(row, col+r);

    // Construct Q explicitly (dorgqr)
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, m, r, r, QR.data(), n, &tau[0]);
    Mat Q1(m, r);
    for (BFInt row = 0; row < m; row++)
        for (BFInt col = 0; col < r; col++)
            Q1(row, col) = QR(row, col);

    // Aa = Q1 * R11 (m x r)(r x r)
    Mat Aa_local(m, r);
    bf_matmult(Q1, R11, Aa_local);

    // T = R11 \ R12
    LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', r, n - r, 
        R11.data(), r, R12.data(), n-r);

    std::stable_sort(pivinv.begin(), pivinv.end(), 
        [&piv](int i1, int i2) { return piv[i1] < piv[i2]; } );

    for (BFInt i = 0; i < n; i++)
    {
        longpiv[i] = (BFInt) piv[i];
        longpivinv[i] = (BFInt) pivinv[i];
    }

    *Aa = new Mat(m, r, Aa_local.getData());
    *Ab = new IDMat(r, n, R12.getData(), longpiv, longpivinv);

    Mat *A_out = nullptr;
    bf_id_decode(*Aa, *Ab, &A_out);
    double err = compare_mats(A, *A_out);
    BFInt maxrank = std::min(m,n);

    Vec x(A_out->getCols());
    Vec yt((*Aa)->getCols());
    Vec xp(A_out->getCols());
    Vec y(A_out->getRows());
    Vec ye(A_out->getRows());
    for (BFInt j = 0; j < x.size(); j++)
        x[j] = (double) j;
    bf_id_apply_b(*Ab, x, yt, xp);
    bf_matvec(*(*Aa), yt, y);
    bf_matvec(A, x, ye);
    double matvecerr = 0.0;
    double matvecnrm = 0.0;
    for (BFInt j = 0; j < y.size(); j++)
    {
        matvecerr += (y[j] - ye[j]) * (y[j] - ye[j]);
        matvecnrm += ye[j] * ye[j];
    }
    matvecerr = sqrt(matvecerr) / sqrt(matvecnrm);

    double materr = compare_mats(R12, (*Ab)->getT());

    // printf("    Encoded block. input: (%d by %d) Rank = %d/%d (%lf). Info = %d, (%.6e, %.6e, %.6e)\n", 
    //     m, n, r, maxrank, (double)r/(double)maxrank, info, err, matvecerr, materr);

    return info;

    // Q: m by n, R: n by n, piv: n
    // [Q,R,piv] = qr(A)
    // r: determine rank using R
    // [~, pivinv] = sort(piv)
    // S = Q(:, 1:r)   * R(1:r, 1:r)
    // T = R(1:r, 1:r) \ R(1:r, r+1:n)

    // Aa = S
    // Ab.T = T
    // Ab.piv = piv
    // Ab.pivinv = pivinv
    // Ab.rank = r
}
void bf_joinandsplit(Mat **B1, Mat **B2, Mat *A1, Mat *A2)
{
    // temp = [A1, A2];
    // [nrows, ~] = size(temp);
    // B1 = temp(1:floor(nrows/2), :);
    // B2 = temp(floor(nrows/2)+1:end, :);

    BFInt inRows  = A1->getRows();
    BFInt inCols1 = A1->getCols();
    BFInt inCols2 = A2->getCols();
    
    BFInt outRows1 = inRows/2;
    BFInt outRows2 = inRows - outRows1;
    BFInt outCols  = inCols1 + inCols2;

    // join A1 and A2
    Mat *temp = new Mat(inRows, inCols1 + inCols2);
    for (BFInt row = 0; row < inRows; row++)
    {
        for (BFInt col = 0; col < inCols1; col++)
            (*temp)(row, col) = (*A1)(row, col);
        for (BFInt col = 0; col < inCols2; col++)
            (*temp)(row, inCols1 + col) = (*A2)(row, col);
    }

    // form B1 and B2
    if (*B1) { delete *B1; *B1 = nullptr; }
    if (*B2) { delete *B2; *B2 = nullptr; }
    *B1 = new Mat(outRows1, outCols);
    *B2 = new Mat(outRows2, outCols);
    // cout << "  forming B1" << endl;
    for (BFInt row = 0; row < outRows1; row++)
        for (BFInt col = 0; col < outCols; col++)
            (**B1)(row, col) = (*temp)(row, col);
    // cout << "  forming B1" << endl;
    for (BFInt row = 0; row < outRows2; row++)
        for (BFInt col = 0; col < outCols; col++)
            (**B2)(row, col) = (*temp)(row + outRows1, col);
}

void bf_encode(double tol, const char* filename, BFMat &bfA, bool debug)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    BFInt L = bfA.getL();
    BFInt P = 1 << L;

    std::vector<MPI_Request> sendRequests(2*6*P);
    std::vector<MPI_Status>  sendStatuses(2*6*P);
    std::vector<MPI_Status>  recvStatuses(2*6*P);

    // Prepare input partition workspace
    std::vector<Mat *> A(2*P);
    for (BFInt i = 0; i < 2*P; i++)
        A[i] = nullptr;

    BFInt nRows, nCols;
    // Read data from file and prepare partitions
    {
        std::vector<double> Adata;
        Mat *A_in;
        if (rank == 0) petsc_binary_read(filename, Adata, &nRows, &nCols);
        if (rank == 0) A_in = new Mat(nRows, nCols, Adata);
        MPI_Bcast(&nRows, 1, MPI_BFINT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nCols, 1, MPI_BFINT, 0, MPI_COMM_WORLD);
        bfA.setGlobalRows(nRows);
        bfA.setGlobalCols(nCols);
        bfA.setInputTolerance(tol);
        
        bf_partition(bfA, A_in, A);
    }
    double totalsize = nRows * nCols * sizeof(double);
    double compressedsize = 0;

    // std::vector<std::vector<double>> send_buffer_1(bfA.getNumberOwned(0));
    // std::vector<std::vector<double>> send_buffer_2(bfA.getNumberOwned(0));
    // std::vector<std::vector<double>> recv_buffer_1(bfA.getNumberOwned(0));
    // std::vector<std::vector<double>> recv_buffer_2(bfA.getNumberOwned(0));

    std::vector<MPI_Datatype> send_type_1(bfA.getNumberOwned(0));
    std::vector<MPI_Datatype> send_type_2(bfA.getNumberOwned(0));
    std::vector<MPI_Datatype> recv_type_1(bfA.getNumberOwned(0));
    std::vector<MPI_Datatype> recv_type_2(bfA.getNumberOwned(0));
    std::vector<BFInt> m1(bfA.getNumberOwned(0));
    std::vector<BFInt> m2(bfA.getNumberOwned(0));
    std::vector<BFInt> n_send(bfA.getNumberOwned(0));

    if (debug && rank == 0) printf("Encoding...\n");
    for (BFInt l = 0; l < L+1; l++)
    {
        if (debug && rank == 0) printf("  Starting layer %d out of %d\n", l, L+1);

        // 2a) ID phase
        for (BFInt j = 0; j < bfA.getNumberOwned(l); j++)
        {
            Mat **Aa_j = bfA.getAaPtrPtr(bfA.getOwned(l, j));
            IDMat **Ab_lj = bfA.getAbPtrPtr(l, bfA.getOwned(l, j));

            auto start_time = Clock::now();
            BFInt info = bf_id_encode(Aa_j, Ab_lj, *(A[bfA.getOwned(l, j)]), tol);
            auto finish_time = Clock::now();
            double elapsed = NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(finish_time-start_time).count();

            BFInt maxrank = std::min(A[bfA.getOwned(l, j)]->getRows(), A[bfA.getOwned(l, j)]->getCols());
            BFInt localrank = (*Ab_lj)->getRank();
            double rankratio = ((double) localrank) / ((double) maxrank);
            if (debug) printf("    [%d] Encoded block %d (%d) out of %d. Rank = %d/%d (%lf). Info = %d, (%d by %d) -> (%d by %d). Elapsed: %lf ms\n", rank, j, bfA.getOwned(l,j), bfA.getNumberOwned(l), localrank, maxrank, rankratio, info, A[bfA.getOwned(l, j)]->getRows(), A[bfA.getOwned(l, j)]->getCols(), localrank, A[bfA.getOwned(l,j)]->getCols(), elapsed);

            bfA.setAbRows(l, bfA.getOwned(l,j), (*Ab_lj)->getTRows());
            bfA.setAbCols(l, bfA.getOwned(l,j), (*Ab_lj)->getTCols() + (*Ab_lj)->getTRows());

            compressedsize += localrank * (A[bfA.getOwned(l, j)]->getCols() - localrank) * sizeof(double);
            compressedsize += 2 * A[bfA.getOwned(l, j)]->getCols() * sizeof(BFInt);
        }

        // 2b) JS phase
        if (l <= L-1)
        {
            if (l > 0)
            {
                for (BFInt j = 0; j < 6*bfA.getNumberOwned(l); j++)
                    MPI_Wait(&sendRequests[j], &sendStatuses[j]);
            }
         
            // printf("Starting send phase\n");
            for (BFInt j = 0; j < bfA.getNumberOwned(l); j++)
            {
                Mat *Aa_j = bfA.getAaPtr(bfA.getOwned(l, j));
                // calculate:
                BFInt k2 = bfA.getPermIdsInv(l, bfA.getOwned(l, j))/2;
                BFInt send_id_1 = 2*k2;
                BFInt send_id_2 = 2*k2+1;
                m1[j] = Aa_j->getRows()/2;
                m2[j] = Aa_j->getRows() - m1[j];
                n_send[j] = Aa_j->getCols();

                MPI_Type_vector(m1[j], n_send[j], n_send[j], MPI_DOUBLE, &send_type_1[j]);
                MPI_Type_commit(&send_type_1[j]);
                MPI_Type_vector(m2[j], n_send[j], n_send[j], MPI_DOUBLE, &send_type_2[j]);
                MPI_Type_commit(&send_type_2[j]);

                // send:
                BFInt comm_offset = 6*bfA.getOwned(l,j) + 12*P*l;
                MPI_Isend(&m1[j], 1, MPI_BFINT, send_id_1 % size, comm_offset, MPI_COMM_WORLD, &sendRequests[6*j]);
                MPI_Isend(&m2[j], 1, MPI_BFINT, send_id_2 % size, comm_offset+1, MPI_COMM_WORLD, &sendRequests[6*j+1]);
                MPI_Isend(&n_send[j], 1, MPI_BFINT, send_id_1 % size, comm_offset+2, MPI_COMM_WORLD, &sendRequests[6*j+2]);
                MPI_Isend(&n_send[j], 1, MPI_BFINT, send_id_2 % size, comm_offset+3, MPI_COMM_WORLD, &sendRequests[6*j+3]);
                // send_buffer_1[j].resize(m1[j] * n_send[j]);
                // for (BFInt row = 0; row < m1[j]; row++)
                //     for (BFInt col = 0; col < n_send[j]; col++)
                //         send_buffer_1[j][row*n_send[j] + col] = (*Aa_j)(row, col);
                // MPI_Isend(&(send_buffer_1[j])[0], m1[j]*n_send[j], MPI_DOUBLE, send_id_1 % size, comm_offset+4, MPI_COMM_WORLD, &sendRequests[6*j+4]);
                // send_buffer_2[j].resize(m2[j] * n_send[j]);
                // for (BFInt row = 0; row < m2[j]; row++)
                //     for (BFInt col = 0; col < n_send[j]; col++)
                //         send_buffer_2[j][row*n_send[j] + col] = (*Aa_j)(row+m1[j], col);
                // MPI_Isend(&(send_buffer_2[j])[0], m2[j]*n_send[j], MPI_DOUBLE, send_id_2 % size, comm_offset+5, MPI_COMM_WORLD, &sendRequests[6*j+5]);

                MPI_Isend(Aa_j->data(0,     0), 1, send_type_1[j], send_id_1 % size, comm_offset+4, MPI_COMM_WORLD, &sendRequests[6*j+4]);
                MPI_Isend(Aa_j->data(m1[j], 0), 1, send_type_2[j], send_id_2 % size, comm_offset+5, MPI_COMM_WORLD, &sendRequests[6*j+5]);
            }
            // printf("Completed send phase, starting recieve phase\n");
            MPI_Barrier(MPI_COMM_WORLD);
            for (BFInt j = 0; j < bfA.getNumberOwned(l); j++)
            {
                BFInt p_offset = bfA.getOwned(l,j) % 2;
                BFInt k1 = bfA.getOwned(l, j)/2;
                BFInt recv_id_1 = bfA.getPermIds(l, 2*k1);
                BFInt recv_id_2 = bfA.getPermIds(l, 2*k1+1);

                BFInt n1, n2;
                BFInt m_recv1, m_recv2;
                BFInt comm_offset1 = 6*recv_id_1 + 12*P*l;
                BFInt comm_offset2 = 6*recv_id_2 + 12*P*l;
                
                MPI_Recv(&m_recv1, 1, MPI_BFINT, recv_id_1 % size, comm_offset1+p_offset,   MPI_COMM_WORLD, &recvStatuses[6*j]);
                MPI_Recv(&m_recv2, 1, MPI_BFINT, recv_id_2 % size, comm_offset2+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+1]);
                MPI_Recv(&n1, 1, MPI_BFINT, recv_id_1 % size, comm_offset1+2+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+2]);
                MPI_Recv(&n2, 1, MPI_BFINT, recv_id_2 % size, comm_offset2+2+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+3]);

                MPI_Type_vector(m_recv1, n1, n1+n2, MPI_DOUBLE, &recv_type_1[j]);
                MPI_Type_commit(&recv_type_1[j]);
                MPI_Type_vector(m_recv1, n2, n1+n2, MPI_DOUBLE, &recv_type_2[j]);
                MPI_Type_commit(&recv_type_2[j]);

                // recv_buffer_1[j].resize(n1*m_recv1);
                // recv_buffer_2[j].resize(n2*m_recv1);
                // MPI_Recv(&(recv_buffer_1[j])[0], n1*m_recv1, MPI_DOUBLE, recv_id_1 % size, comm_offset1+4+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+4]);
                // MPI_Recv(&(recv_buffer_2[j])[0], n2*m_recv1, MPI_DOUBLE, recv_id_2 % size, comm_offset2+4+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+5]);

                if (A[bfA.getOwned(l,j)]) { delete A[bfA.getOwned(l,j)]; A[bfA.getOwned(l,j)] = nullptr; }
                A[bfA.getOwned(l,j)] = new Mat(m_recv1, n1 + n2);

                MPI_Recv(A[bfA.getOwned(l,j)]->data(0,  0), 1, recv_type_1[j], recv_id_1 % size, comm_offset1+4+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+4]);
                MPI_Recv(A[bfA.getOwned(l,j)]->data(0, n1), 1, recv_type_2[j], recv_id_2 % size, comm_offset2+4+p_offset, MPI_COMM_WORLD, &recvStatuses[6*j+5]);

                // for (BFInt row = 0; row < m_recv1; row++)
                //     for (BFInt col = 0; col < n1; col++)
                //         (*A[bfA.getOwned(l,j)])(row, col) = recv_buffer_1[j][row*n1 + col];
                // for (BFInt row = 0; row < m_recv1; row++)
                //     for (BFInt col = 0; col < n2; col++)
                //         (*A[bfA.getOwned(l,j)])(row, col+n1) = recv_buffer_2[j][row*n2 + col];
            }
            // printf("Completed recieve phase\n");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    
    for (BFInt j = 0; j < bfA.getNumberOwned(L); j++)
    {
        Mat *Aa_j = bfA.getAaPtr(bfA.getOwned(L,j));
        bfA.setAaRows(bfA.getOwned(L,j), Aa_j->getRows());
        bfA.setAaCols(bfA.getOwned(L,j), Aa_j->getCols());
        compressedsize += Aa_j->getRows() * Aa_j->getCols() * sizeof(double);
    }

    for (BFInt l = 0; l <= L; l++)
    {
        for (BFInt j = 0; j < 2*P; j++)
        {
            BFInt lrows, lcols;
            if (rank == j % size)
            {
                lrows = bfA.getAbRows(l,j);
                lcols = bfA.getAbCols(l,j);
            }
            MPI_Bcast(&lrows, 1, MPI_BFINT, j % size, MPI_COMM_WORLD);
            MPI_Bcast(&lcols, 1, MPI_BFINT, j % size, MPI_COMM_WORLD);
            bfA.setAbRows(l,j,lrows);
            bfA.setAbCols(l,j,lcols);
        }
    }
    for (BFInt j = 0; j < 2*P; j++)
    {
        BFInt lrows, lcols;
        if (rank == j % size)
        {
            lrows = bfA.getAaRows(j);
            lcols = bfA.getAaCols(j);
        }
        MPI_Bcast(&lrows, 1, MPI_BFINT, j % size, MPI_COMM_WORLD);
        MPI_Bcast(&lcols, 1, MPI_BFINT, j % size, MPI_COMM_WORLD);
        bfA.setAaRows(j,lrows);
        bfA.setAaCols(j,lcols);
    }
    
    double totalCompressedSize;
    MPI_Allreduce(&compressedsize, &totalCompressedSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double compressionRatio = totalCompressedSize / totalsize;
    bfA.setCompressionRatio(compressionRatio);
    MPI_Barrier(MPI_COMM_WORLD);
}