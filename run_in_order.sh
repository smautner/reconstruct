#!/bin/bash

REPEATS=100

execute () {
    # sed '/SGE_TASK_ID/s/$/'"$STRING"'/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	JOBID=$(qsub -V -t 1-$REPEATS runall_custom_sge.sh "$STRING"| sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
	echo "Current Task: $JOBID"
	while true; do
        STATE=$(qstat)	# qstat
	    if [[ ! $STATE =~ $JOBID ]]; then
            break
        fi
	    sleep 10
	done
	# sed 's/SGE_TASK_ID.*/SGE_TASK_ID/' runall_custom_sge.sh > _tmp.sh_ && mv -- _tmp.sh_ runall_custom_sge.sh
	echo "$JOBID" >> results.txt
	echo "$STRING" >> results.txt
	python reconstruct.py report >> results.txt
	echo "Finished $JOBID at $(date)"
}

echo "Start: $(date)"
# Cipselector 1: 100 200 400 800
for CIPK in 600; do
    STRING=" -n --cipselector_option 1 --cipselector_k $CIPK"
	execute
	mkdir -p results/cipsel1_$CIPK
	mv .res/* results/cipsel1_$CIPK/
done

# Cipselector 2: (Default) 1 5 10 15 20
for CIPK in 15; do
    STRING=" -n --cipselector_option 2 --cipselector_k $CIPK"
	execute
	mkdir -p results/cipsel2_$CIPK
	mv .res/* results/cipsel2_$CIPK/
done

# Normalization:
for NORM in ''; do
    STRING=" $NORM"
	execute
	mkdir -p results/nonorm$NORM
	mv .res/* results/nonorm$NORM/
done

# Pareto Options:
for PARETO in 'default' 'random' 'greedy' 'pareto_only' 'all'; do
    STRING=" -n --pareto_option $PARETO"
	execute
	mkdir -p results/pareto$PARETO
	mv .res/* results/pareto$PARETO/
done

# Contextsizes/Thickness: 
for CONTEXTSIZE in 1 2; do
    STRING=" -n --context_size $CONTEXTSIZE"
	execute
	mkdir -p results/contextsize$CONTEXTSIZE
	mv .res/* results/contextsize$CONTEXTSIZE/
done
