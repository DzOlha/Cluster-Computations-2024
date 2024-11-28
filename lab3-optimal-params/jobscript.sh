#!/bin/bash
echo NPROCS: $SLURM_NPROCS
#source ../env/bin/activate

##### FIRST argument is granularity, SECOND argument is broadcast_rate
granularity=$1
broadcast_rate=$2
monte_index=$3

time mpirun python3.10 stats_3.py --gran=$granularity --bcast_rate=$broadcast_rate --monte_index=$monte_index