#!/bin/bash
#$ -cwd
#$ -l h_vmem=3330M
#$ -pe smp 6
#$ -R y
#$ -o logs/$JOB_ID.o_$TASK_ID
#$ -e logs/$JOB_ID.e_$TASK_ID
python reconstruct.py chem $SGE_TASK_ID
