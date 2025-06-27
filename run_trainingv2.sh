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
pop_size=200
num_parents_mating=100
mut_prob1=0.25
mut_prob2=0.1

for num in "${nums[@]}"
do
    echo "Running script for num: $num"
    python3 GA_optimizer_v2.py --num_processes "$num" --population_size "$pop_size" --num_parents_mating "$num_parents_mating" --mutation_probability "$mut_prob1" "$mut_prob2"
done

