// backward operations
void bf_joinandsplit_inverse(Mat **A1, Mat **A2, Mat *B1, Mat *B2, BFInt k)
{
    
    // A1 = temp(:, 1:k);
    // A2 = temp(:, k+1:end);

    BFInt inRows1 = B1->getRows();
    BFInt inRows2 = B2->getRows();
    BFInt inCols  = B1->getCols();

    BFInt outRows  = inRows1 + inRows2;
    BFInt outCols1 = k;
    BFInt outCols2 = inCols - k;

    // temp = [B1; B2];
    Mat temp(inRows1 + inRows2, inCols);
    for (BFInt row = 0; row < inRows1; row++)
        for (BFInt col = 0; col < inCols; col++)
            temp(row, col) = (*B1)(row, col);
    for (BFInt row = 0; row < inRows2; row++)
        for (BFInt col = 0; col < inCols; col++)
            temp(row + inRows1, col) = (*B2)(row, col);

    // form A1 and A2
    if (*A1) { delete *A1; *A1 = nullptr; }
    if (*A2) { delete *A2; *A2 = nullptr; }
    *A1 = new Mat(outRows, outCols1);
    *A2 = new Mat(outRows, outCols2);
    for (BFInt row = 0; row < outRows; row++)
    {
        for (BFInt col = 0; col < outCols1; col++)
            (**A1)(row, col) = temp(row, col);
        for (BFInt col = 0; col < outCols2; col++)
            (**A2)(row, col) = temp(row, col + outCols1);
    }
}
void bf_id_decode(Mat *Aa, IDMat *Ab, Mat **A)
{
    if (*A) { delete *A; (*A) = nullptr; }

    BFInt m = Aa->getRows();
    BFInt r = Aa->getCols();
    BFInt n = Ab->getT().getCols() + r;

    (*A) = new Mat(m, n);

    for (BFInt row = 0; row < m; row++)
    {
        for (BFInt col = 0; col < r; col++)
        {
            (**A)(row, Ab->piv(col)-1) = (*Aa)(row, col);
        }
    }
    Mat R(m, n - r);
    bf_matmult(*Aa, Ab->getT(), R);
    for (BFInt row = 0; row < m; row++)
    {
        for (BFInt col = 0; col < n - r; col++)
        {
            (**A)(row, Ab->piv(r + col)-1) = R(row, col);
        }
    }
}
void bf_partition_inverse(BFInt P, std::vector<Mat *> &A, Mat &A_out)
{
    BFInt nRows = A_out.getRows();
    BFInt nCols = A_out.getCols();
    BFInt gRow = 0;
    BFInt gCol = 0;
    BFInt lRows, lCols;
    BFInt count = 0;
    for (BFInt bCol = 0; bCol < P; bCol++)
    {
        gRow = 0;
        for (BFInt bRow = 0; bRow < 2; bRow++)
        {
            //BFInt b = 2*bCol + bRow;
            lRows = A[count]->getRows();
            lCols = A[count]->getCols();
            
            for (BFInt row = 0; row < lRows; row++)
                for (BFInt col = 0; col < lCols; col++)
                    A_out(gRow + row, gCol + col) = (*A[count])(row, col);
            
            gRow += lRows;
            count++;
        }
        gCol += lCols;
    }
}
void bf_decode(BFInt P, BFInt L, Mat &A, 
    std::vector<std::vector<BFInt>> &perm_ids,
    std::vector<Mat *> &Ap, 
    std::vector<std::vector<Mat *>> &Aa, 
    std::vector<std::vector<IDMat *>> &Ab)
{
    printf("Decoding...\n");
    for (BFInt ll = L+1; ll > 0; ll--)
    {
        BFInt l = ll - 1;

        //printf("  Starting layer %d out of %d\n", l, L+1);

        // 3a) ID evaluation phase
        for (BFInt j = 0; j < 2*P; j++)
        {
            
            bf_id_decode(Aa[l][j], Ab[l][j], &Ap[j]);
            //printf("    Decoded IDMat %d out of %d\n", j, 2*P);
        }

        // 3b) Inverse JS phase
        if (l > 0)
        {
            for (BFInt j = 0; j < P; j++)
            {
                BFInt j1 = 2*j;
                BFInt j2 = 2*j + 1;
                BFInt k1 = perm_ids[l-1][j1];
                BFInt k2 = perm_ids[l-1][j2];
                bf_joinandsplit_inverse(&Aa[l-1][k1], &Aa[l-1][k2], 
                    Ap[j1], Ap[j2], Ab[l-1][k1]->getRank());
            }
        }
    }
    bf_partition_inverse(P, Ap, A);
}