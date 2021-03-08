#PBS -l nodes=1:ppn=4
#PBS -l walltime=04:00:00
#PBS -l mem=4gb
#PBS -S /bin/bash
#PBS -o logs/${PBS_JOBID}.o_${PBS_ARRAYID}
#PBS -e logs/${PBS_JOBID}.e_${PBS_ARRAYID}


source /beegfs/work/workspace/ws/fr_mh595-conda-0/conda/etc/profile.d/conda.sh
conda activate binenv
cd $PBS_O_WORKDIR

python reconstruct.py --taskid ${PBS_ARRAYID}
