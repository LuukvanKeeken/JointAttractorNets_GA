#!/bin/bash

cd /project_antwerp/code/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad
pip install brian2
pip install lmfit

nums=(4)
size=100
num_cpu=4

for num in "${nums[@]}"
do
    echo "Running script for num: $num"
    python3 parallel_processing_test4.py "$num" "$size" "$num_cpu"
done