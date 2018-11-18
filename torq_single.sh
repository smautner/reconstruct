#!/bin/bash
#PBS -l nodes=1:ppn=1,mem=16gb
#PBS -N mabob
#PBS -q "short"
##PBS -l walltime=24:00:00
#PBS -d /home/fr/fr_fr/fr_sm1105/code/reconstruct
#PBS -o /home/fr/fr_fr/fr_sm1105/code/reconstruct/OUT
#PBS -j oe



cd /home/fr/fr_fr/fr_sm1105/code/reconstruct


#seq $1 $2 | parallel --results .log -j 20 python3 reconstruct.py 
python3 reconstruct.py $1

