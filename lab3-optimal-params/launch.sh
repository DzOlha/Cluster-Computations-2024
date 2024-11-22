#!/bin/bash

#### must get NUMBER_OF_PROCESSES as parameters, and generate granularity and broadcast_rate


###### FIRST argument is number_of_processes
number_of_processes=$1
granularity=5
broadcast_rate=10

sbatch -n $number_of_processes --mem-per-cpu=10G -p scit5 jobscript.sh $granularity $broadcast_rate