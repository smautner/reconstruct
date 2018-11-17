#!/bin/bash
#PBS -l pmem=6000mb
#PBS -l nodes=1:ppn=20 
#PBS -N mabob
#PBS -q "short"

#PBS -d /home/fr/fr_fr/fr_sm1105/code/reconstruct
#PBS -o /home/fr/fr_fr/fr_sm1105/code/reconstruct/OUT
#PBS -j oe


cd /home/fr/fr_fr/fr_sm1105/code/reconstruct

rm ".res/"*

seq $1 $2 | parallel --results .log -j 20 python3 reconstruct.py 


#qsub -q short runmoab.fish -F "1 20"  
