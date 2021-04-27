#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --partition=soc-kp
#SBATCH --account=soc-kp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 28
#SBATCH -J emm3d
#SBATCH -o logs/square4k.%J.out

export EXPNAME=square4k
export SOLVETYPE=EigenMM2D
export INPUTDIR="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/EigenMM/experiments/mcompress/input"
export OUTPUTDIR="/scratch/kingspeak/serial/u0450449/eigenbasis/data"
export OPTIONSDIR="/uufs/chpc.utah.edu/common/home/u0450449/Fractional/EigenMM/experiments/mcompress/options"

export NUMNODES=1
export TASKSPERNODE=28
export ENCTASKS=16
export ENCTASKSPERNODE=16
export VERBOSE=1

export L=2
export TOL=1e-4

source $MKLROOT/bin/mklvars.sh intel64

# generate EigenMM options file
python ./set_options.py $TASKSPERNODE $OPTIONSDIR $OUTPUTDIR $EXPNAME

# solve for eigenbasis
mpirun -np $SLURM_NTASKS ../../build/EigenMMSolve $INPUTDIR/$EXPNAME $OPTIONSDIR/${EXPNAME}_options.xml

# compress eigenbasis
mpirun -np $ENCTASKS -perhost $ENCTASKSPERNODE ../../build/MCompressEncode $OUTPUTDIR/$EXPNAME $L $TOL