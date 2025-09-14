#!/bin/bash

#script to be run server-side

# install requirements, then execute script with given arguments

cd /project_antwerp/code/temp2/JointAttractorNets_GA
# pip install wfdb
python -m pip install -U pip
pip install pyparsing==3.2.1
pip3 install pandas==2.2.3
pip install pygad==3.4.0
pip install brian2==2.8.0.4
pip install lmfit==1.3.3
pip install tqdm==4.67.1



#for 1 run:
python3 "Optimization Models/gridSearch_optimizer_v2.py"

