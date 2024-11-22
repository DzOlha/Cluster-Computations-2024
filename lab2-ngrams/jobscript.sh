#!/bin/bash
echo NPROCS: $SLURM_NPROCS
source ../env/bin/activate
time mpirun python3.10 stats.py