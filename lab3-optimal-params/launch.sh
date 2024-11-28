#!/bin/bash

TOTAL_MEMORY=64

# Number of Monte Carlo simulations
MONTE_CARLO_ITERATIONS=20

# Range for the number of processes, granularity, and broadcast_rate
MIN_PROCESSES=1
MAX_PROCESSES=4
MIN_GRANULARITY=1
MAX_GRANULARITY=20
MIN_BROADCAST_RATE=1
MAX_BROADCAST_RATE=5

# Function to generate a random number between a given range
generate_random() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

for i in $(seq 1 $MONTE_CARLO_ITERATIONS); do
    # Randomly select number of processes, granularity, and broadcast rate
    number_of_processes=$(generate_random $MIN_PROCESSES $MAX_PROCESSES)
    granularity=$(generate_random $MIN_GRANULARITY $MAX_GRANULARITY)
    broadcast_rate=$(generate_random $MIN_BROADCAST_RATE $MAX_BROADCAST_RATE)

    MEM_PER_PROCESS=$((TOTAL_MEMORY / number_of_processes))

    # Submit the job
    sbatch -n $number_of_processes --mem-per-cpu=${MEM_PER_PROCESS}M -p main-partition jobscript.sh $granularity $broadcast_rate $i
done