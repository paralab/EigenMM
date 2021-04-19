#include <vector>
#include <assert.h>

class Vec
{
private:
    std::vector<double> mData;
public:
    Vec() : mData(1) {}
    Vec(BFInt n) : mData(n) {}
    Vec(std::vector<double> &data) : mData(data) {}
    double& operator[](BFInt i)        { return mData[i]; }
    double  operator[](BFInt i)  const { return mData[i]; }
    BFInt  size()                const { return mData.size(); }
    double* begin() { return &mData[0]; }
    void resize(BFInt l) { mData.resize(l); }
};

Vec operator+(Vec const &u, Vec const &v) 
{
    assert(u.size() == v.size());
    Vec sum(u.size());
    for(BFInt i = 0; i < u.size(); i++)
    {
        sum[i] = u[i] + v[i];
    }
    return sum;
}

class Mat
{
private:
    BFInt mRows;
    BFInt mCols;
    std::vector<double> mData;
public:
    Mat() : mRows(0), mCols(0), mData(1) {}
    Mat(BFInt rows, BFInt cols) : mRows(rows), mCols(cols), mData(rows*cols) {}
    Mat(BFInt rows, BFInt cols, std::vector<double> &data) : mRows(rows), mCols(cols), mData(data) {}
    //~Mat() { std::cout << "Deleting " << mRows << " by " << mCols << " matrix." << std::endl; }
    double& operator()(BFInt i, BFInt j) 
    { 
        return mData[i*mCols + j]; 
    }
    double operator()(BFInt i, BFInt j) const
    {
        return mData[i*mCols + j];
    }
    BFInt getRows() { return mRows; }
    BFInt getCols() { return mCols; }
    BFInt getSize() { return mData.size(); }
    std::vector<double> &getData() { return mData; }
    double* data() { return &mData[0]; }
    double* data(BFInt i, BFInt j) { return &mData[i*mCols + j]; }
};

void bf_matmult(Mat &A, Mat &B, Mat &C)
{
    if (A.getSize() > 0 && B.getSize() > 0 && C.getSize() > 0)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            (int) A.getRows(), (int) B.getCols(), (int) A.getCols(), 
            1.0, A.data(), (int) A.getCols(), B.data(), (int) B.getCols(), 
            0.0, C.data(), (int) C.getCols());
}

void bf_matvec(Mat &A, Vec &x, Vec &y)
{
    // if (A.getCols() > 0)
    //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         A.getRows(), 1, A.getCols(),
    //         1.0, A.data(), A.getCols(), x.begin(), 1,
    //         0.0, y.begin(), 1);
    if (A.getCols() > 0)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, (int) A.getRows(), (int) A.getCols(),
            1.0, A.data(), (int) A.getCols(), x.begin(), 1, 0.0, y.begin(), 1);
    else
    {
        for (BFInt i = 0; i < y.size(); i++)
            y[i] = 0.0;
    }
}

void bf_matvec_add(Mat *A, Vec &x, Vec &y)
{
    // y = A*x + y
    if (A->getCols() > 0)
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
            (int) A->getRows(), (int) A->getCols(),
            1.0, A->data(), (int) A->getCols(), x.begin(), 1,
            1.0, y.begin(), 1);
}

void print(Mat &M)
{
    std::cout << "Rows: " << M.getRows() << ", Cols: " << M.getCols() << ", Size: " << M.getSize() << std::endl;
    for (BFInt i = 0; i < M.getRows(); i++)
    {
        for (BFInt j = 0; j < M.getCols(); j++)
            std::cout << M(i,j) << " ";
        std::cout << std::endl;
    }
}

void print(Mat &M, BFInt rows, BFInt cols, BFInt row0, BFInt col0)
{
    for (BFInt i = 0; i < rows; i++)
    {
        for (BFInt j = 0; j < cols; j++)
            std::cout << M(i+row0,j+col0) << " ";
        std::cout << std::endl;
    }
}

class IDMat
{
private:
    BFInt mRank;
    Mat mT;
    std::vector<BFInt> mPiv;
    std::vector<BFInt> mPivInv;
public:
    IDMat(BFInt rank, BFInt cols, std::vector<double> &data, 
        std::vector<BFInt> &piv, std::vector<BFInt> &pivinv)
        : mRank(rank),
          mT(rank, cols-rank, data),
          mPiv(piv),
          mPivInv(pivinv) {}
    //~IDMat() { printf("Deleting %d rank IDMat.\n", mRank); }
    BFInt getRank() { return mRank; }
    BFInt getTRows() { return mT.getRows(); }
    BFInt getTCols() { return mT.getCols(); }
    BFInt piv(BFInt i) { return mPiv[i]; }
    BFInt pivinv(BFInt i) { return mPivInv[i]; }
    BFInt* pivdata() { return &mPiv[0]; }
    BFInt* pivinvdata() { return &mPivInv[0]; }
    Mat &getT() { return mT; }
    Mat *getTPtr() { return &mT; }

};

void bf_id_apply_b(IDMat *Ab, Vec &x, Vec &y, Vec &xp)
{
    BFInt m = Ab->getRank();
    BFInt n = Ab->getT().getCols() + m;

    for (BFInt i = 0; i < n - m; i++)
        xp[i] = x[Ab->piv(m+i)-1]; 
    for (BFInt i = 0; i < m; i++)
        y[i] = x[Ab->piv(i)-1];
    bf_matvec_add(Ab->getTPtr(), xp, y);
}

class BFMat
{
private:
    int mRank;
    int mSize;
    BFInt mL;
    BFInt mP;
    BFInt mGlobalRows;
    BFInt mGlobalCols;
    double mCompressionRatio;
    double mInputTolerance;
    std::vector<std::vector<BFInt>> mPermIds;
    std::vector<std::vector<BFInt>> mPermIdsInv;
    std::vector<BFInt> mAaRows;
    std::vector<BFInt> mAaCols;
    std::vector<std::vector<BFInt>> mAbRows;
    std::vector<std::vector<BFInt>> mAbCols;
    std::vector<std::vector<BFInt>> mAbRanks;
    std::vector<Mat *> mAa;
    std::vector<std::vector<IDMat *>> mAb;
    std::vector<std::vector<BFInt>> mOwned;
    std::vector<std::vector<BFInt>> mOwnedMap;
    
    std::vector<Vec> mXp;
    std::vector<Vec> mYp;
    std::vector<Vec> mXppiv;
    
    std::vector<BFInt> mX0;
    std::vector<BFInt> mY0;
    std::vector<MPI_Request *> mSendRequests;
    std::vector<MPI_Request *> mGatherRequests;
    std::vector<MPI_Request *> mRecvRequests;
    std::vector<MPI_Status *> mRecvStatuses;
    std::vector<MPI_Status *> mSendStatuses;


    void perm_ids_init()
    {
        BFInt len, nsegs, id0;
        mPermIds.resize(mL);
        mPermIdsInv.resize(mL);
        for (BFInt l = 0; l < mL; l++)
        {
            mPermIds[l].resize(2*mP);
            mPermIdsInv[l].resize(2*mP);
            len = (2*mP) >> l;
            nsegs = (2*mP) / len;
            id0 = 0;
            for(BFInt s = 0; s < nsegs; s++)
            {
                for (BFInt i = 0; i < len/2; i++)
                    mPermIds[l][id0 + i] = id0 + i*2;
                for (BFInt i = len/2; i < len; i++)
                    mPermIds[l][id0 + i] = id0 + (i-len/2)*2 + 1;
                id0 += len;
            }
        }   

        std::vector<BFInt> range(2*mP);
        std::vector<BFInt> perms;
        for (BFInt l = 0; l < mL; l++)
        {
            perms = mPermIds[l];
            for (BFInt j = 0; j < 2*mP; j++)
                range[j] = j;
            std::stable_sort(range.begin(), range.end(), 
                [&perms](BFInt i1, BFInt i2) { return perms[i1] < perms[i2]; } );
            std::copy(range.begin(), range.end(), mPermIdsInv[l].begin());
        }
    }
public:
    BFMat() : mL(0), mP(1), mGlobalRows(1), mGlobalCols(1), 
        mCompressionRatio(-1.0) 
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mSize);
    }
    BFMat(BFInt L) { init(L); }
    void init(BFInt L)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mSize);

        mL = L;
        mP = 1 << L;
        mGlobalRows = 0;
        mGlobalCols = 0;
        mCompressionRatio = -1.0;

        perm_ids_init();

        mOwned.resize(mL+1);

        mAaRows.resize(2*mP);
        mAaCols.resize(2*mP);
        mAa.resize(2*mP);

        mAbRows.resize(mL+1);
        mAbCols.resize(mL+1);
        mAbRanks.resize(mL+1);
        mAb.resize(mL+1);
        for (BFInt l = 0; l <= mL; l++)
        {
            mAbRows[l].resize(2*mP);
            mAbCols[l].resize(2*mP);
            mAbRanks[l].resize(2*mP);
            mAb[l].resize(2*mP);
        }
    }

    BFInt getGlobalRows() { return mGlobalRows; }
    BFInt getGlobalCols() { return mGlobalCols; }
    BFInt getL() { return mL; }
    double getCompressionRatio() { return mCompressionRatio; }
    double getInputTolerance() { return mInputTolerance; }

    BFInt* getAaRowData() { return &mAaRows[0]; }
    BFInt* getAaColData() { return &mAaCols[0]; }
    BFInt* getAbRowData(BFInt l) { return &(mAbRows[l])[0]; }
    BFInt* getAbColData(BFInt l) { return &(mAbCols[l])[0]; }
    double* getAaData(BFInt j) { return mAa[j]->data(); }
    double* getAbTData(BFInt l, BFInt j) { return mAb[l][j]->getT().data(); }
    double* getXpData(BFInt j) { return mXp[j].begin(); }
    BFInt* getAbPivData(BFInt l, BFInt j) { return mAb[l][j]->pivdata(); }
    BFInt* getAbPivInvData(BFInt l, BFInt j) { return mAb[l][j]->pivinvdata(); }

    BFInt getAaRows(BFInt j) { return mAaRows[j]; }
    BFInt getAaCols(BFInt j) { return mAaCols[j]; }
    BFInt getAbRows(BFInt l, BFInt j) { return mAbRows[l][j]; }
    BFInt getAbCols(BFInt l, BFInt j) { return mAbCols[l][j]; }
    BFInt getPermIds(BFInt l, BFInt j) { return mPermIds[l][j]; }
    BFInt getPermIdsInv(BFInt l, BFInt j) { return mPermIdsInv[l][j]; }
    BFInt getNumberOwned(BFInt l) { return mOwned[l].size(); }
    BFInt getOwned(BFInt l, BFInt j) { return mOwned[l][j]; }
    BFInt getOwnedMap(BFInt l, BFInt j) { return mOwnedMap[l][j]; }
    BFInt getY0(BFInt i) { return mY0[i]; }
    BFInt getX0(BFInt i) { return mX0[i]; }

    void setGlobalRows(BFInt v) { mGlobalRows = v; }
    void setGlobalCols(BFInt v) { mGlobalCols = v; }
    void setCompressionRatio(double v) { mCompressionRatio = v; }
    void setInputTolerance(double v) { mInputTolerance = v; }
    void setAaRows(BFInt j, BFInt v) { mAaRows[j] = v; }
    void setAaCols(BFInt j, BFInt v) { mAaCols[j] = v; }
    void setAbRows(BFInt l, BFInt j, BFInt v) { mAbRows[l][j] = v; }
    void setAbCols(BFInt l, BFInt j, BFInt v) { mAbCols[l][j] = v; }

    void setAa(BFInt j, BFInt nrows, BFInt ncols, std::vector<double> &data)
    {
        mAa[j] = new Mat(nrows, ncols, data);
    }
    void setAb(BFInt l, BFInt j, BFInt nrows, BFInt ncols, std::vector<double> &data, std::vector<BFInt> &piv, std::vector<BFInt> &pivinv)
    {
        mOwned[l].push_back(j);
        mAb[l][j] = new IDMat(nrows, ncols, data, piv, pivinv);
    }
    void setOwned(BFInt l, BFInt j)
    {
        mOwned[l].push_back(j);
    }
    void setOwnedMap(std::vector<std::vector<BFInt>> &map)
    {
        mOwnedMap.resize(mL+1);
        for (BFInt l = 0; l <= mL; l++)
            mOwnedMap[l] = map[l];
    }

    void initializeXp(BFInt max_m)
    {
        mXp.resize(mXp.size()+1);
        mXp[mXp.size()-1].resize(max_m);
        mXppiv.resize(mXppiv.size()+1);
        mXppiv[mXppiv.size()-1].resize(max_m);
    }
    void initializeYp(BFInt max_n)
    {
        mYp.resize(mYp.size()+1);
        mYp[mYp.size()-1].resize(max_n);
    }

    Mat** getAaPtrPtr(BFInt j) { return &mAa[j]; }
    IDMat** getAbPtrPtr(BFInt l, BFInt j) { return &mAb[l][j]; }
    Mat* getAaPtr(BFInt j) { return mAa[j]; }
    IDMat* getAbPtr(BFInt l, BFInt j) { return mAb[l][j]; }

    void initializeWorkspace()
    {
        mX0.resize(mP+1);
        mX0[0] = 0;
        for (BFInt i = 1; i < mP; i++)
            mX0[i] = mX0[i-1] + mAbCols[0][2*(i-1)];
        mX0[mP] = mGlobalCols;

        mY0.resize(2*mP+1);
        mY0[0] = 0;
        for (BFInt i = 1; i < 2*mP; i++)
            mY0[i] = mY0[i-1] + mAaRows[i-1];
        mY0[2*mP] = mGlobalRows;

        mSendRequests.resize(4*mP);
        mGatherRequests.resize(4*mP);
        mRecvRequests.resize(4*mP);
        mRecvStatuses.resize(4*mP);
        mSendStatuses.resize(4*mP);
        for (BFInt i = 0; i < 4*mP; i++)
        {
            mSendRequests[i] = new MPI_Request();
            mGatherRequests[i] = new MPI_Request();
            mRecvRequests[i] = new MPI_Request();
            mRecvStatuses[i] = new MPI_Status();
            mSendStatuses[i] = new MPI_Status();
        }
    }

    void timed_apply(Vec &x, Vec &y, double *out_waiting_time, double *out_mult_time, double *out_communication_time, bool debug)
    {
	int testflag;
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        double wait_time = 0.0;
        double mult_time = 0.0;
        double communication_time = 0.0;
        auto start_time = Clock::now();
        auto stop_time = Clock::now();
        for (BFInt l = 0; l < mL; l++)
        {
            BFInt totalcost = 0;
            for (BFInt j = 0; j < mOwned[l].size(); j++)
            {
                if (l > 0)
                {
                    stop_time = Clock::now();
                    communication_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
                    start_time = Clock::now();
                    MPI_Wait(mRecvRequests[2*j],   mRecvStatuses[2*j]);
                    MPI_Wait(mRecvRequests[2*j+1], mRecvStatuses[2*j+1]);
                    MPI_Wait(mSendRequests[2*j],   mSendStatuses[2*j]);
                    MPI_Wait(mSendRequests[2*j+1], mSendStatuses[2*j+1]);
                }

                // apply Ab[l][j] to mXp[j] to get mYp[j]
                // int cost = (mAb[l][mOwned[j]]->getT().getRows() + 1) * mAb[l][mOwned[j]]->getT().getCols();
                // totalcost += cost;
                // printf("[Process %d]: Cost for applying Ab[%d][%d] is %d\n", mRank, l, mOwned[j], totalcost);
                if (l == 0 && j != 0)
                {
                    stop_time = Clock::now();
                    communication_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
                    start_time = Clock::now();
                }
                else
                {
                    stop_time = Clock::now();
                    wait_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
                    start_time = Clock::now();
                }
                
                bf_id_apply_b(mAb[l][mOwned[l][j]], mXp[j], mYp[j], mXppiv[j]);
                stop_time = Clock::now();
                //if (debug) printf("[Process %d] Multiplying Ab[%d][%d] (%d by %d), (%.4lf ms)\n", rank, l, mOwned[l][j], mAb[l][mOwned[l][j]]->getRank(), mAb[l][mOwned[l][j]]->getT().getCols(), NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count());
                mult_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
                start_time = Clock::now();

                // communicate partial results
                BFInt k1 = mOwned[l+1][j]/2;
                BFInt recv_id_1 = mPermIds[l][2*k1];
                BFInt recv_id_2 = mPermIds[l][2*k1+1];

                BFInt k2 = mPermIdsInv[l][mOwned[l][j]]/2;
                BFInt send_id_1 = 2*k2;
                BFInt send_id_2 = 2*k2+1;
                
                MPI_Isend(&mYp[j][0], mAbRows[l][mOwned[l][j]], MPI_DOUBLE, 
                    mOwnedMap[l+1][send_id_1], mOwned[l][j], MPI_COMM_WORLD, 
                    (mSendRequests[2*j]));
                MPI_Isend(&mYp[j][0], mAbRows[l][mOwned[l][j]], MPI_DOUBLE, 
                    mOwnedMap[l+1][send_id_2], mOwned[l][j], MPI_COMM_WORLD, 
                    (mSendRequests[2*j+1]));
                MPI_Irecv(&mXp[j][0], mAbRows[l][recv_id_1], MPI_DOUBLE,
                    mOwnedMap[l][recv_id_1], recv_id_1, MPI_COMM_WORLD, 
                    (mRecvRequests[2*j]));
                MPI_Irecv(&mXp[j][mAbRows[l][recv_id_1]], 
                    mAbRows[l][recv_id_2], MPI_DOUBLE, mOwnedMap[l][recv_id_2], 
                    recv_id_2, MPI_COMM_WORLD, 
                    (mRecvRequests[2*j+1]));

		MPI_Test(mSendRequests[2*j], &testflag, mSendStatuses[2*j]);
                MPI_Test(mSendRequests[2*j+1], &testflag, mSendStatuses[2*j+1]);
                MPI_Test(mRecvRequests[2*j], &testflag, mRecvStatuses[2*j]);
                MPI_Test(mRecvRequests[2*j+1], &testflag, mRecvStatuses[2*j+1]);

            }
            // printf("[Process %d]: Total cost for layer %d is %d\n", mRank, l, totalcost);
        }
        for (BFInt j = 0; j < mOwned[mL].size(); j++)
        {
            // make sure mXp[j] and mYp[j] are done with their sends or recieves
            stop_time = Clock::now();
            communication_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
            start_time = Clock::now();
            MPI_Wait(mRecvRequests[2*j],   mRecvStatuses[2*j]);
            MPI_Wait(mRecvRequests[2*j+1], mRecvStatuses[2*j+1]);
            MPI_Wait(mSendRequests[2*j],   mSendStatuses[2*j]);
            MPI_Wait(mSendRequests[2*j+1], mSendStatuses[2*j+1]);
            stop_time = Clock::now();
            wait_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
            start_time = Clock::now();

            // apply Ab[L][j]
            bf_id_apply_b(mAb[mL][mOwned[mL][j]], mXp[j], mYp[j], mXppiv[j]);

            // apply Aa[j]
            bf_matvec((*(mAa[mOwned[mL][j]])), mYp[j], mXp[j]);

            stop_time = Clock::now();
            mult_time += NS2MS * std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time-start_time).count();
            start_time = Clock::now();

            // send final answer to root
            //MPI_Isend(mXp[j].begin(), mAaRows[mOwned[j]], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, (mGatherRequests[j]));
        }

        out_waiting_time[0] = wait_time;
        out_mult_time[0] = mult_time;
        out_communication_time[0] = communication_time;
    }

    void apply(Vec &x, Vec &y)
    {
        // all vectors have all of x on entry, root vector has all of y on exit

        // each process obtains the portions of x for which they are reponsible
        // for (int j = 0; j < mOwned.size(); j++)
        //     std::copy(x.begin()+mX0[mOwned[j]/2], x.begin()+mX0[mOwned[j]/2+1], mXp[j].begin());

        for (BFInt l = 0; l < mL; l++)
        {
            BFInt totalcost = 0;
            for (BFInt j = 0; j < mOwned[l].size(); j++)
            {
                if (l > 0)
                {
                    // make sure mXp[j] and mYp[j] are done with their sends or recieves
                    MPI_Wait(mRecvRequests[2*j],   mRecvStatuses[2*j]);
                    MPI_Wait(mRecvRequests[2*j+1], mRecvStatuses[2*j+1]);
                    MPI_Wait(mSendRequests[2*j],   mSendStatuses[2*j]);
                    MPI_Wait(mSendRequests[2*j+1], mSendStatuses[2*j+1]);
                }

                // apply Ab[l][j] to mXp[j] to get mYp[j]
                // int cost = (mAb[l][mOwned[j]]->getT().getRows() + 1) * mAb[l][mOwned[j]]->getT().getCols();
                // totalcost += cost;
                // printf("[Process %d]: Cost for applying Ab[%d][%d] is %d\n", mRank, l, mOwned[j], totalcost);
                bf_id_apply_b(mAb[l][mOwned[l][j]], mXp[j], mYp[j], mXppiv[j]);

                // communicate partial results
                BFInt k1 = mOwned[l+1][j]/2;
                BFInt recv_id_1 = mPermIds[l][2*k1];
                BFInt recv_id_2 = mPermIds[l][2*k1+1];

                BFInt k2 = mPermIdsInv[l][mOwned[l][j]]/2;
                BFInt send_id_1 = 2*k2;
                BFInt send_id_2 = 2*k2+1;
                
                MPI_Isend(&mYp[j][0], mAbRows[l][mOwned[l][j]], MPI_DOUBLE, 
                    mOwnedMap[l+1][send_id_1], mOwned[l][j], MPI_COMM_WORLD, 
                    (mSendRequests[2*j]));
                MPI_Isend(&mYp[j][0], mAbRows[l][mOwned[l][j]], MPI_DOUBLE, 
                    mOwnedMap[l+1][send_id_2], mOwned[l][j], MPI_COMM_WORLD, 
                    (mSendRequests[2*j+1]));
                MPI_Irecv(&mXp[j][0], mAbRows[l][recv_id_1], MPI_DOUBLE,
                    mOwnedMap[l][recv_id_1], recv_id_1, MPI_COMM_WORLD, 
                    (mRecvRequests[2*j]));
                MPI_Irecv(&mXp[j][mAbRows[l][recv_id_1]], 
                    mAbRows[l][recv_id_2], MPI_DOUBLE, mOwnedMap[l][recv_id_2], 
                    recv_id_2, MPI_COMM_WORLD, 
                    (mRecvRequests[2*j+1]));
            }
            // printf("[Process %d]: Total cost for layer %d is %d\n", mRank, l, totalcost);
        }
        for (BFInt j = 0; j < mOwned[mL].size(); j++)
        {
            // make sure mXp[j] and mYp[j] are done with their sends or recieves
            MPI_Wait(mRecvRequests[2*j],   mRecvStatuses[2*j]);
            MPI_Wait(mRecvRequests[2*j+1], mRecvStatuses[2*j+1]);
            MPI_Wait(mSendRequests[2*j],   mSendStatuses[2*j]);
            MPI_Wait(mSendRequests[2*j+1], mSendStatuses[2*j+1]);

            // apply Ab[L][j]
            bf_id_apply_b(mAb[mL][mOwned[mL][j]], mXp[j], mYp[j], mXppiv[j]);

            // apply Aa[j]
            bf_matvec((*(mAa[mOwned[mL][j]])), mYp[j], mXp[j]);

            // send final answer to root
            //MPI_Isend(mXp[j].begin(), mAaRows[mOwned[j]], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, (mGatherRequests[j]));
        }

        // if (mRank == 0)
        // {
        //     for (int j = 0; j < 2*mP; j++)
        //         MPI_Irecv(y.begin() + mY0[j], mAaRows[j], MPI_DOUBLE, 
        //             j % mSize, 0, MPI_COMM_WORLD, (mRecvRequests[j]));
        //     for (int j = 0; j < 2*mP; j++)
        //         MPI_Wait((mRecvRequests[j]), (mRecvStatuses[j]));
        // }
        //MPI_Barrier(MPI_COMM_WORLD);
    }
};
