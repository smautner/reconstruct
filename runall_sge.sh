#!/bin/bash
#$ -cwd
#$ -l h_vmem=1G
#$ -pe smp 4
#$ -R y
#$ -o logs/$JOB_ID.o_$TASK_ID
#$ -e logs/$JOB_ID.e_$TASK_ID
python reconstruct.py $SGE_TASK_ID
echo $JOB_ID > lastjobid.tmp