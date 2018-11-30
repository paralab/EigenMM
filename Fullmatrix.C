#include "eigen_maj.h"

void FullMatrixnewnew_3D(eigen_maj &eigSolver, Bsystem *B, Bsystem *Bp, Element_List *U, char trip){

    int modess=iparam("MODES");

    int nel=B->nel;

    int nsolve=B->nsolve;

    int nsolve1=B->nsolve+nel*(modess-2)*(modess-2)*(modess-2);

    //B->solvemapw is the map I use for local to global mapping.

    int i,j,k;

    //  for(i=0;i<nsolve1;i++)
    //    dzero(nsolve1,A1[i],1);

    // for(i=0;i<nsolve1;i++)
    // {
    //  for(j=0;j<nsolve1;j++)
    //	printf("%f  ", A1[i][j]);
    //  printf("\n");
    //}
    //printf("\nabove is the A1 matrix in drive.C mynode()=%d!U->flist[0]->id=%d\n",mynode(),U->flist[0]->id);

    int asize;
    int wsize;

    //int eDIM=2;

    double *aloc;
    double *wloc;

    int cnt;

    int **bmap=imatrix(0,B->nel-1,0,U->flist[0]->Nmodes-1);
    // printf("\nU->flist[0]->Nmodes=%d\n",U->flist[0]->Nmodes);

    for(i=0;i<B->nel;i++)
    icopy(U->flist[0]->Nmodes,B->bmapw[i],1,bmap[i],1);

    int nbl;

    Element *E;
    // double *sign = B->signchangew;
    double *sign=NULL;
    // sign=B->signchange;
    // if(!B->signchangew){
    //printf("\nhere is before setup_signchange!\n");
    setup_signchangew(U,B);
    //}

    //for(i=0;i<gNverts;i++)
    // printf("%f  ", B->signchangew[i]);
    // printf("\nabove is the sign vector in EIGENS.C gNverts=%d!\n",gNverts);

    //for(i=0;i<B->nel;i++)
    //{
    //  for(j=0;j<U->flist[0]->Nmodes;j++)
    //	printf("%d  ", bmap[i][j]);
    //  printf("\n");
    //}
    //printf("\nabove is the bmapw vector in EIGENS.C gNverts!\n");

    /*
    for(k = 0; k < nel; ++k){
        E=U->flist[k];
        nbl   = U->flist[k]->Nbmodes;
        asize = U->flist[k]->Nbmodes;
        wsize = U->flist[k]->Nbmodes+(modess-2)*(modess-2)*(modess-2);
        //printf("\nhere is right after constructing matrix1!nsolve=%d nel=%d k=%d wsize=%d\n",nsolve,nel,k,wsize);
        //aloc = B->Gmat->a[U->flist[k]->geom->id];
        aloc = B->Gmat->aold[U->flist[k]->geom->id];
        //printf("\nhere is right after constructing matrix1!nsolve=%d nel=%d k=%d wsize=%d\n",nsolve,nel,k,wsize);
        if(trip=='s')
            wloc = Bp->Gmat->wold[U->flist[k]->geom->id];
        else if(trip=='m')
            wloc = B->Gmat->wold[U->flist[k]->geom->id];
        // {
        //for(i=0;i<wsize*(wsize+1)/2;i++)
        //  wloc[i] = B->Gmat->wold[U->flist[k]->geom->id][i]-Bp->Gmat->wold[U->flist[k]->geom->id][i];
        //}
        //printf("\nhere is right after wloc before print\n");
        //for(i=0;i<wsize*(wsize+1)/2;i++)
        //printf("%f  ", wloc[i]);
        //printf("\nabove is the wloc matrix in drive.C U->flist[k]->geom->id=%d k=%d!\n\n\n",U->flist[k]->geom->id,k);
        //for(i=0;i<asize*(asize+1)/2;i++)
        //printf("%f  ", aloc[i]);
        // printf("\nabove is the aloc matrix in drive.C U->flist[k]->geom->id=%d k=%d!\n\n\n",U->flist[k]->geom->id,k);


        //for(i=0;i<wsize*(wsize+1)/2;i++)
        //printf("%f  ", wloc[i]);
        //printf("\nabove is the wloc matrix in drive.C U->flist[k]->geom->id=%d k=%d!\n",U->flist[k]->geom->id,k);
        // for(i=0;i<nsolve1;i++)
        //{
        //	for(j=0;j<nsolve1;j++)
        //	  printf("%f  ", A1[i][j]);
        //	printf("\n");
        //}
        //printf("\nabove is the A1 matrix in drive.C mynode()=%d!U->flist[0]->id=%d\n",mynode(),U->flist[0]->id);


        //we need to reconstruct bmap to bmap1 so that we can get together all the data!!!
        cnt = 0;
        for(i = 0; i < wsize; ++i){
            if(i<E->Nbmodes&&bmap[k][i]>=nsolve)
                //if(0)
            {
                //printf("\ncnt=%d\n",cnt);
                cnt+=i+1;
                //printf("\ncnt=%d\n",cnt);
            }
                // if(i<E->Nverts&&E->vert[i].solve==0)
                //cnt += i+1;
                //else if(i<E->Nbmodes&&i>=E->Nverts&&E->edge[(i-E->Nverts)/(modess-2)].solve==0)
                //cnt += i+1;
            else
            {

                //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                if(bmap[k][i] < nsolve1){
                    //printf("\nhere is right 1\n");
                    for(j = 0; j < i; ++j)
                        if(i<E->Nbmodes&&bmap[k][j] < nsolve){
                            //	printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                            //	printf("\nhere is right 2,i=%d j=%d k=%d bmap[k][i]=%d bmap[k][j]=%d B->signchangewsign[j]=%f wloc[cnt]=%f\n",i,j,k,bmap[k][i],bmap[k][j],B->signchangew[j],wloc[cnt]);
                            A1[bmap[k][i]][bmap[k][j]] += B->signchangew[i+k*wsize]*B->signchangew[j+k*wsize]*wloc[cnt];
                            //A1[bmap[k][i]][bmap[k][j]] += 0;
                            //printf("\nhere is right 3\n");
                            A1[bmap[k][j]][bmap[k][i]] += B->signchangew[i+k*wsize]*B->signchangew[j+k*wsize]*wloc[cnt++];
                            //printf("\nhere is right 4\n");
                            //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                        }
                        else if(i>=E->Nbmodes&&j>=E->Nbmodes&&bmap[k][j] < nsolve1)
                        {
                            //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                            // printf("\nhere is right 2,i=%d j=%d k=%d bmap[k][i]=%d bmap[k][j]=%d B->signchangewsign[j]=%f wloc[cnt]=%f cnt=%d\n",i,j,k,bmap[k][i],bmap[k][j],B->signchangew[j],wloc[cnt],cnt);
                            A1[bmap[k][i]][bmap[k][j]] += B->signchangew[i+k*wsize]*B->signchangew[j+k*wsize]*wloc[cnt];
                            //A1[bmap[k][i]][bmap[k][j]] += 0;
                            //printf("\nhere is right 3\n");
                            A1[bmap[k][j]][bmap[k][i]] += wloc[cnt++];
                            //printf("\nhere is right 4\n");
                            //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                        }

                        else if(i>=E->Nbmodes&&bmap[k][j] < nsolve&&bmap[k][j]!=bmap[k][i])
                        {
                            //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                            //printf("\nhere is right 2,i=%d j=%d k=%d bmap[k][i]=%d bmap[k][j]=%d B->signchangewsign[j]=%f wloc[cnt]=%f cnt=%d\n",i,j,k,bmap[k][i],bmap[k][j],B->signchangew[j],wloc[cnt],cnt);
                            A1[bmap[k][i]][bmap[k][j]] += B->signchangew[i+k*wsize]*B->signchangew[j+k*wsize]*wloc[cnt];
                            //A1[bmap[k][i]][bmap[k][j]] += 0;
                            //printf("\nhere is right 3\n");
                            A1[bmap[k][j]][bmap[k][i]] += B->signchangew[i+k*wsize]*B->signchangew[j+k*wsize]*wloc[cnt++];
                            //printf("\nhere is right 4\n");
                            //printf("\nhere is right 0A1[bmap[k][i]][bmap[k][i]]%f\n",A1[bmap[k][i]][bmap[k][i]]);
                        }

                        else
                            ++cnt;
                    //do diagonal
                    //printf("\nhere is right 2,i=%d j=%d k=%d bmap[k][i]=%d bmap[k][j]=%d B->signchangewsign[j]=%f wloc[cnt]=%f %f cnt=%d\n",i,j,k,bmap[k][i],bmap[k][j],B->signchangew[j],wloc[cnt],A1[bmap[k][i]][bmap[k][i]],cnt);
                    A1[bmap[k][i]][bmap[k][i]] += wloc[cnt++];
                    //	printf("\nhere is right 2,i=%d j=%d k=%d bmap[k][i]=%d bmap[k][j]=%d B->signchangewsign[j]=%f wloc[cnt]=%f\n",i,j,k,bmap[k][i],bmap[k][j],B->signchangew[j],wloc[cnt]);
                }
                else//as above here is for lower packed matrix
                    cnt += i+1;
            }
        }
        //sign += wsize;


    }
*/

    Mat            A1,B1;           /* matrices */
    EPS            eps;             /* eigenproB1lem solver context */
    ST             st;
    KSP            ksp;
    EPSType        type;
    PetscReal      tol;
    Vec            xr,xi,*Iv,*Cv;
    PetscInt       nev,maxit,its,lits,nconv,nini=0,ncon=0, ncv, mpd;
//    PetscInt       i;
    char           filename[PETSC_MAX_PATH_LEN];
    PetscViewer    viewer;
    PetscBool      flg,evecs,ishermitian,terse;
//    PetscErrorCode ierr;

    int n = 100; // todo: set matrix size here.

    SlepcInitialize(0, nullptr, nullptr, nullptr);

    MatCreate(PETSC_COMM_WORLD, &A1); // todo: set the correct MPI_Comm here.
    MatSetSizes(A1, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetFromOptions(A1);
    MatSetUp(A1);

    for(k = 0; k < nel; ++k){
        E=U->flist[k];
        wsize = U->flist[k]->Nbmodes+(modess-2)*(modess-2)*(modess-2);
        if(trip=='s') {
            wloc = Bp->Gmat->wold[U->flist[k]->geom->id];
        }else if(trip=='m') {
            wloc = B->Gmat->wold[U->flist[k]->geom->id];
        }

        //we need to reconstruct bmap to bmap1 so that we can get together all the data!!!
        cnt = 0;
        for(i = 0; i < wsize; ++i){
            if(i<E->Nbmodes&&bmap[k][i]>=nsolve) {
                cnt += i+1;
            } else {
                if(bmap[k][i] < nsolve1){
                    for(j = 0; j < i; ++j) {
                        if (i < E->Nbmodes && bmap[k][j] < nsolve) {
                            eigSolver.setA(bmap[k][i], bmap[k][j], B->signchangew[i + k * wsize] * B->signchangew[j + k * wsize] * wloc[cnt]);
                            eigSolver.setA(bmap[k][j], bmap[k][i], B->signchangew[i + k * wsize] * B->signchangew[j + k * wsize] * wloc[cnt++]);
                        } else if (i >= E->Nbmodes && j >= E->Nbmodes && bmap[k][j] < nsolve1) {
                            eigSolver.setA(bmap[k][i], bmap[k][j], B->signchangew[i + k * wsize] * B->signchangew[j + k * wsize] * wloc[cnt]);
                            eigSolver.setA(bmap[k][j], bmap[k][i], wloc[cnt++], ADD_VALUES);
                        } else if (i >= E->Nbmodes && bmap[k][j] < nsolve && bmap[k][j] != bmap[k][i]) {
                            eigSolver.setA(bmap[k][i], bmap[k][j], B->signchangew[i + k * wsize] * B->signchangew[j + k * wsize] * wloc[cnt]);
                            eigSolver.setA(bmap[k][j], bmap[k][i], B->signchangew[i + k * wsize] * B->signchangew[j + k * wsize] * wloc[cnt++]);
                        } else {
                            ++cnt;
                        }
                    }

                    // diagonal
                    eigSolver.setA(bmap[k][i], bmap[k][i], wloc[cnt++]);

                } else { //as above here is for lower packed matrix
                    cnt += i + 1;
                }
            }
        }
    }

}

