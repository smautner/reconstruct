#!/bin/bash
#$ -cwd
#$ -l h_vmem=2G
#$ -pe smp 4
#$ -R y
#$ -o logs/$JOB_ID.o_$TASK_ID
#$ -e logs/$JOB_ID.e_$TASK_ID
python reconstruct.py chem $SGE_TASK_ID
