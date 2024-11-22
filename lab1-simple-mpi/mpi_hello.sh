#!/bin/bash
#SBATCH --job-name=mpi_hello_job
#SBATCH --output=mpi_hello_output.txt
#SBATCH --ntasks=4
#SBATCH --time=00:05:00

mpirun -n 4 mpi_hello