#!/bin/bash

# Total memory available for the job (15GB in bytes)
TOTAL_MEMORY=15360  # 15GB in MB because the Wiki dataset has 13 GB

for i in $(seq 1 5 100); do
  # Calculate memory per process (in MB)
  MEM_PER_PROCESS=$((TOTAL_MEMORY / i))

  # Submit the job with the calculated memory per process
  sbatch -n $i --mem-per-cpu=${MEM_PER_PROCESS}M -p scit5 jobscript.sh
done
