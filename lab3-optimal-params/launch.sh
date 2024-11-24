#!/bin/bash

TOTAL_MEMORY=15360

# Number of Monte Carlo simulations
MONTE_CARLO_ITERATIONS=200

# Range for the number of processes, granularity, and broadcast_rate
MIN_PROCESSES=1
MAX_PROCESSES=100
MIN_GRANULARITY=1
MAX_GRANULARITY=20
MIN_BROADCAST_RATE=1
MAX_BROADCAST_RATE=50

# Function to generate a random number between a given range
generate_random() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

# Store the best combination and performance
best_time=999999999
best_combination=""

# Initialize a JSON object to store results
results="{"


for i in $(seq 1 $MONTE_CARLO_ITERATIONS); do
    # Randomly select number of processes, granularity, and broadcast rate
    number_of_processes=$(generate_random $MIN_PROCESSES $MAX_PROCESSES)
    granularity=$(generate_random $MIN_GRANULARITY $MAX_GRANULARITY)
    broadcast_rate=$(generate_random $MIN_BROADCAST_RATE $MAX_BROADCAST_RATE)

    MEM_PER_PROCESS=$((TOTAL_MEMORY / number_of_processes))

    # Submit the job and get the result
    job_output=$(sbatch -n $number_of_processes --mem-per-cpu=${MEM_PER_PROCESS}M -p scit5 jobscript.sh $granularity $broadcast_rate)
    job_id=$(echo $job_output | awk '{print $4}')  # Get job ID from the sbatch output

    # Optionally wait for the job to finish or collect results at a later point
    # We can monitor the job's progress or fetch its output for evaluation
    job_status=$(sacct -j $job_id --format="State,Elapsed" -n | tail -n 1)
    job_state=$(echo $job_status | awk '{print $1}')
    job_elapsed=$(echo $job_status | awk '{print $2}')

    # Evaluate the job (here we're assuming the time is in the format HH:MM:SS)
    if [[ $job_state == "COMPLETED" ]]; then
        # Parse elapsed time (HH:MM:SS -> seconds)
        IFS=: read -r hours minutes seconds <<< "$job_elapsed"
        elapsed_seconds=$((10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))

        # Store the results for this combination in the JSON object
        results+="'$i': {'processes': $number_of_processes, 'granularity': $granularity, 'broadcast_rate': $broadcast_rate, 'time': $elapsed_seconds},"

        # Update best combination if the current job performed better (shorter time)
        if [[ $elapsed_seconds -lt $best_time ]]; then
            best_time=$elapsed_seconds
            best_combination="Processes: $number_of_processes, Granularity: $granularity, Broadcast Rate: $broadcast_rate"
        fi
    fi
done

# Remove the trailing comma from the JSON object
results="${results%,}"

# Close the JSON object
results+="}"

# Save the results to a JSON file
echo "$results" > results.json

# After Monte Carlo simulations, print the best combination and time
echo "Best combination found: $best_combination with time: $best_time seconds"
