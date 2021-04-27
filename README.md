# EigenMM + MCompress

EigenMM is a scalable spectrum slicing eigenvalue solver that extends PETSc and SLEPc. MCompress is a hierarchical compression library for compressing the resulting eigenbasis matrices.

To solve a generalized eigenvalue problem, PETSc should be installed with mumps, metis, parmetis, and scalapack.\
To do that, when installing PETSc pass the following parameters:\
./configure --with-cxx-dialect=C++11 --download-metis --download-parmetis --download-scalapack --download-mumps

EigenMM also uses TinyXML for reading in an options file.

An example of how to use the full EigenMM + MCompress pipeline can be found in the `experiments` folder.

---

## Installation
An example install script has been provided named `example_install.sh` that shows the cmake variables that need to be set. EigenMM + MCompress requires installs of PETESc, SLEPc, TinyXML, and Intel MKL. 

---

## MCompress Options

- `L`: The partitioning size paramter. Input matrix is partitioned into 2 rows and 2^L columns of blocks.
- `tol`: The tolerance parameter for rank estimation of partitions.

---

## EigenMM Solver Options
There are a number of options that can be supplied to EigenMM if the default settings are not sufficient.

### Communicator Options

- `nodesperevaluator`: A node is a collection of processes that are all located on the same physical machine. This option determines how many nodes make up a single evaluator. Typically, this should just be 1, meaning that 1 node = 1 evaluator.
- `nevals`: The number of eigenvalues to solve for. A negative value here will solve for all eigenvalues.
- `taskspernode`: The number of MPI tasks per node.

### Partitioning Options

- `nk`: The maximum number of iterations fot the global partitioning stage.
- `nb`: The maximum number of iterations for the partition balancing stage.
- `splitmaxiters`: The maximum number of iterations for balancing a pair of intervals.
- `splittol`: The tolerance determine how accurately a pair of intervals are balanced.
- `raditers`: The maximum number of iterations to be done when approximating the spectral radius.
- `radtol`: The tolerance determining how accurately the spectral radius is computed. The accuracy doesn't need to be very precise to capture all eigenvalues.
- `L`, `R`: The endpoints of the interval containing all eigenvalues you are intending to compute.

### Save/Print Options
- `debug`: Enables debug messages.
- `save_eigenvalues`, `eigenvalues_filename`: Set save to 1 and provide a filename to save the computed eigenvalues to file.
- `save_eigenbasis`, `eigenbasis_filename`: Set save to 1 and provide a filename to save the comptued eigenbasis matrix to file.
