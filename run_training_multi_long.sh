#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/temp/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad==3.4.0
pip install brian2
pip install lmfit
python -m pip install -U matplotlib


num_processes=(16)
pop_size=(200)
num_parents_mating=(100)
mut_prob1=(0.5)
mut_prob2=(0.3)
num_gens=(500)
input_on=(0.5)
input_off=(0.5)

initial_ranges_mex=(0.05 0.3 0.05 0.3 0.5 1.0 -1.2 -0.3)  # Mexican hat profile ranges

for i in "${!num_processes[@]}"; do
    num_proc="${num_processes[$i]}"
    pop_size="${pop_size[$i]}"
    num_parents_mating="${num_parents_mating[$i]}"
    mut_prob1="${mut_prob1[$i]}"
    mut_prob2="${mut_prob2[$i]}"
    num_gen="${num_gens[$i]}"

    echo "Running script for settings: $num_proc processes, $pop_size population, $num_parents_mating parents mating, $mut_prob1 mutation prob1, $mut_prob2 mutation prob2, $num_gen generations"
    python3 GA_optimizer_cleaned_multi.py --num_processes "$num_proc" --population_size "$pop_size" --num_parents_mating "$num_parents_mating" --mutation_probability "$mut_prob1" "$mut_prob2" --num_generations "$num_gen" --initial_ranges_mex "${initial_ranges_mex[@]}" --input_on "$input_on" --input_off "$input_off"
done

