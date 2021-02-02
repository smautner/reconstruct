#!/bin/bash
echo -n > results.txt
mkdir -p results
# Cipselector 1:
for CIPK in 100 200 400 800; do
    STRING=" --cipselector_option 1 --cipselector_k $CIPK"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-2 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/cipsel1_$CIPK
	mv .res/* results/cipsel1_$CIPK/
done
# Cipselector 2: (Default)
for CIPK in 1 5 10 15 20; do
    STRING=" --cipselector_option 2 --cipselector_k $CIPK"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-100 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/cipsel2_$CIPK
	mv .res/* results/cipsel2_$CIPK/
done
# Normalization:
for NORM in ' --use_nomralization' ''; do
    STRING="$NORM"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-100 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/norm$NORM
	mv .res/* results/norm$NORM/
done
# Pareto Options:
for PARETO in 'default' 'random' 'greedy' 'pareto_only' 'all'; do
    STRING=" --pareto_option $PARETO"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-100 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/pareto$PARETO
	mv .res/* results/pareto$PARETO/
done
# Coresizes/Radii:
for CORESIZES in '0 1 2' '2' '0 2 4'; do
    STRING=" --core_sizes $CORESIZES"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-100 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/coresize$CORESIZES
	mv .res/* results/coresize$CORESIZES/
done
# Contextsizes/Thickness: 
for CONTEXTSIZE in 0.5 1 2; do
    STRING=" --context_size $CONTEXTSIZE"
    sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-100 runall_custom_sge.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
    STATE=`qstat`	# qstat
#	echo $STATE
	if [[ ! $STATE =~ $JOBID ]]; then
    break
    fi
	sleep 10
	done
	sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	mkdir -p results/contextsize$CONTEXTSIZE
	mv .res/* results/contextsize$CONTEXTSIZE/
done
