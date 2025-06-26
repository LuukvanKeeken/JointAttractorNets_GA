#!/bin/bash

cd /project_antwerp/code/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad
pip install brian2
pip install lmfit

nums=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
size=100
num_cpu=1

for num in "${nums[@]}"
do
    echo "Running script for num: $num"
    python3 parallel_processing_test4.py "$num" "$size" "$num_cpu"
done