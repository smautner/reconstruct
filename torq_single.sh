#!/bin/bash
#PBS -l nodes=1:ppn=20,mem=120gb
#PBS -N mabob
#PBS -q "short"
##PBS -l walltime=24:00:00
#PBS -d /home/fr/fr_fr/fr_sm1105/code/reconstruct_chem
#PBS -o /home/fr/fr_fr/fr_sm1105/code/reconstruct_chem/OUT
#PBS -j oe



cd /home/fr/fr_fr/fr_sm1105/code/reconstruct_chem


python3 reconstruct.py chem $1



# this was for parallel stuff.. just chills in the q ... 
##PBS -l nodes=1:ppn=20,mem=127gb
#seq $1 $2 | parallel --results .log -j 20 python3 reconstruct.py 
