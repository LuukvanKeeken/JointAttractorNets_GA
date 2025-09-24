#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/temp/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip install numpy==2.2.2
pip install pyparsing==3.2.3
pip3 install pandas==2.2.3
pip install pygad==3.4.0
pip install brian2==2.8.0.4
pip install lmfit==1.3.3
python -m pip install -U matplotlib


num_processes=(16)
pop_size=(16)
num_parents_mating=(8)
mut_prob1=(0.5)
mut_prob2=(0.3)
num_gens=(2)
con_prof="cosine"

initial_ranges_cos=(0.01 0.5 -2.0 -0.01)

for i in "${!num_processes[@]}"; do
    num_proc="${num_processes[$i]}"
    pop_size="${pop_size[$i]}"
    num_parents_mating="${num_parents_mating[$i]}"
    mut_prob1="${mut_prob1[$i]}"
    mut_prob2="${mut_prob2[$i]}"
    num_gen="${num_gens[$i]}"



    echo "Running script for settings: $num_proc processes, $pop_size population, $num_parents_mating parents mating, $mut_prob1 mutation prob1, $mut_prob2 mutation prob2, $num_gen generations"
    python3 GA_optimizer_cleaned_multi_faithful.py --num_processes "$num_proc" --population_size "$pop_size" --num_parents_mating "$num_parents_mating" --mutation_probability "$mut_prob1" "$mut_prob2" --num_generations "$num_gen" --initial_ranges_cos "${initial_ranges_cos[@]}" --connectivity_profile "$con_prof"
done

