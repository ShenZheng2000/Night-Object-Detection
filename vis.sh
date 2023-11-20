#!/bin/bash

# Define the array of first parameters
first_params=(1 2 4 8 16 32 64 128 256)

# Define the array of second parameters
second_params=(0.5 0.75 1.0)

# Loop over each combination of parameters and run the python script
for first in "${first_params[@]}"; do
    for second in "${second_params[@]}"; do
        python vis.py "$first" "$second"
    done
done