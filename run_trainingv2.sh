#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad
pip install brian2
pip install lmfit


nums=(16)
size=100

for num in "${nums[@]}"
do
    echo "Running script for num: $num"
    python3 GA_optimizer_v2.py --num_processes "$num" --population_size "$size"
done

