#!/bin/bash
#$ -cwd
#$ -l h_vmem=2G
#$ -M mautner@cs.uni-freiburg.de
#$ -pe smp 8
#$ -R y
#$ -o /home/mautner/JOBZ/reconstr_o/$JOB_ID.o_$TASK_ID
#$ -e /home/mautner/JOBZ/reconstr_e/$JOB_ID.e_$TASK_ID

##mkdir -p /home/mautner/JOBZ/reconstr_o/
##mkdir -p /home/mautner/JOBZ/reconstr_e/
python reconstruct.py $SGE_TASK_ID
#qsub -V -t 1-720  runall_sge.sh

