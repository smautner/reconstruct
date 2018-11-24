#!/bin/bash
#$ -cwd
#$ -l h_vmem=5G
#$ -M mautner@cs.uni-freiburg.de
#$ -pe smp 24
#$ -R y
#$ -o /home/mautner/JOBZ/reconstr_o_c/$JOB_ID.o_$TASK_ID
#$ -e /home/mautner/JOBZ/reconstr_e_c/$JOB_ID.e_$TASK_ID

##mkdir -p /home/mautner/JOBZ/reconstr_o_c/
##mkdir -p /home/mautner/JOBZ/reconstr_e_c/
python reconstruct.py chem $SGE_TASK_ID
#qsub -V -t 1-100  chemrecon.sh

