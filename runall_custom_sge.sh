#!/bin/bash
#$ -cwd
#$ -l h_vmem=1G
#$ -pe smp 4
#$ -R y
#$ -o logs/$JOB_ID.o_$TASK_ID
#$ -e logs/$JOB_ID.e_$TASK_ID
echo "$1"
python reconstruct.py --taskid $SGE_TASK_ID $1
