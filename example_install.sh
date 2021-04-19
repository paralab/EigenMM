rm -rf build
mkdir build
cd build

cmake \
-DCMAKE_CXX_COMPILER=icpc \
-DCMAKE_C_COMPILER=icc \
-DPETSC_ARCH="c-opt" \
-DPETSC_ROOT="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/nektar/build/ThirdParty/petsc-3.11.4" \
-DSLEPC_ROOT="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/nektar/build/ThirdParty/slepc-3.11.2" \
-DMKL_INCLUDE_DIRS="/uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2018.1.163/linux/mkl/include" \
-DMKL_LIBRARIES="/uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib" \
-DTINYXML_INCLUDE="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/tinyxml-2.6.2" \
-DTINYXML_LIB="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/tinyxml-2.6.2/build/libtinyxml.a" \
..

make
