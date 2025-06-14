#!/bin/bash

cd /project_antwerp/code/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad
pip install brian2

nums=(1 2 3)

for num in "${nums[@]}"
do
    echo "Running script for num: $num"
    python3 parallel_processing_test2.py "$num"
done