#!/bin/bash

# Total memory available for the job (15GB in bytes)
TOTAL_MEMORY=64  # 15GB in MB because the Wiki dataset has 13 GB

for i in $(seq 2 1 4); do
  # Submit the job with the calculated memory per process
  sbatch -n $i --mem-per-cpu=${TOTAL_MEMORY}M -p main-partition jobscript.sh
done
