#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/temp2/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip3 install pandas
pip install pygad
pip install brian2
pip install lmfit
pip install tqdm



#for 1 run:
python3 gridSearch_optimizer_v2.py

