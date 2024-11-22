#!/bin/bash
for i in $seq(50 5 100):
do
  sbatch -n $i --mem-per-cpu=15GB -p scit5 jobscript.sh
done