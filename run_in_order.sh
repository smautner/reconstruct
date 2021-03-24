#!/bin/bash

source /beegfs/work/workspace/ws/fr_mh595-conda-0/conda/etc/profile.d/conda.sh
conda activate binenv

REPEATS=50 ### Change to 100 for Normal or 250 for Chem

execute () {    ######## Make sure to change filename in first sed command to 'chem_runall_binac.sh' or just 'runall_binac.sh'

    sed '/reconstruct.py/s/$/'"$STRING"'/' runall_binac.sh > .run_$RESPREFIX.sh
    JOBID=$(qsub -q short -t 1-$REPEATS .run_$RESPREFIX.sh | sed -r 's/^[^0-9]*([0-9]+).*$/\1/')
    echo "Current Task: $JOBID : $STRING"
    echo "$JOBID : $STRING" >> results.txt
}

######IF sed is not used:    JOBID=$(qsub -q short -t 1-$REPEATS runall_binac.sh "$STRING"| sed -r 's/^[^0-9]*([0-9]+).*$/\1/')

report () {
    echo "$STRING" >> results.txt
    python reconstruct.py report --resprefix $RESPREFIX >> results.txt
}

reportchem () {
    echo "$STRING" >> results.txt
    python reconstruct.py reportchem --resprefix $RESPREFIX >> results.txt
}

pass () {
    echo "Passing on: $RESPREFIX"
}


echo "Start: $(date)"
## Parameter Optimization
CIPSELECTOR=1
for PARETO in 'random' 'greedy' 'pareto_only' 'all' 'default'; do
    for NORMALIZATION in 1; do
        for CIPK in 500 1000; do
            for DECOMPRADIUS in 1 2; do
                for CONTEXTSIZE in 1 2; do
                    for MINCOUNT in 1; do
                        for SIZELIMITER in 1; do
                            RESPREFIX="$CIPSELECTOR-$CIPK-$CONTEXTSIZE-$MINCOUNT-$SIZELIMITER-$NORMALIZATION-$DECOMPRADIUS-$PARETO"
                            STRING=" --cipselector_option $CIPSELECTOR --pareto_option $PARETO --cipselector_k $CIPK --context_size $CONTEXTSIZE --min_count $MINCOUNT --graph_size_limiter $SIZELIMITER --use_normalization $NORMALIZATION --max_decompose_radius $DECOMPRADIUS --resprefix $RESPREFIX"
                            ##report ##  Replace this with report/execute/pass
                        done
                    done
                done
            done
        done
    done
done
## Chemset Comparison
for CONTEXTSIZE in 1 2; do
    for CIPK in 100 200 300 400; do
        RESPREFIX="cipK-$CIPK-contextsize-$CONTEXTSIZE"
        STRING=" --pareto_option 'default' --core_sizes 0 1 2 3 --context_size $CONTEXTSIZE --cipselector_k $CIPK --resprefix $RESPREFIX"
#        reportchem
    done
done

## Artificial Comparison
CIPSELECTOR=2
for CONTEXTSIZE in 1 2; do
    RESPREFIX="coresizes-012-contextsize-$CONTEXTSIZE"
    STRING=" --cipselector_option $CIPSELECTOR --core_sizes 0 1 2 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
    RESPREFIX="coresizes-01-contextsize-$CONTEXTSIZE"
    STRING=" --cipselector_option $CIPSELECTOR --core_sizes 0 1 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
    RESPREFIX="coresizes-0-contextsize-$CONTEXTSIZE"
    STRING=" --cipselector_option $CIPSELECTOR --core_sizes 0 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
done

## Comparison of Average Productions ##
    CIPSELECTOR=0
    CIPK=1000
    RESPREFIX="cipsel0_$CIPK"
    STRING=" --cipselector_option $CIPSELECTOR --cipselector_k $CIPK --resprefix $RESPREFIX"
#    report
    CIPSELECTOR=1
    CIPK=100
    RESPREFIX="cipsel1_$CIPK"
    STRING=" --cipselector_option $CIPSELECTOR --cipselector_k $CIPK --resprefix $RESPREFIX"
#    report
    CIPSELECTOR=2
    CIPK=10
    RESPREFIX="cipsel2_$CIPK"
    STRING=" --cipselector_option $CIPSELECTOR --cipselector_k $CIPK --resprefix $RESPREFIX"
#    report
##########################

## Cipselector 1: 100 200 400 800
for CIPK in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200; do
#for CIPK in 10 30 50 70 90; do
    RESPREFIX="cipsel1_$CIPK"
    STRING=" --cipselector_option 1 --cipselector_k $CIPK --resprefix $RESPREFIX"
#    report
done

for KEEP in 12 30 60; do
    for CIPK in 50 100 250; do
        RESPREFIX="keep-$KEEP-cipk-$CIPK"
        STRING=" --keepgraphs $KEEP --cipselector_option 1 --cipselector_k $CIPK --resprefix $RESPREFIX"
#        report
    done
done


## Cipselector 2: 1 5 10 15 20 #### REMOVED 100 FOR CHEMSETS
for CIPK in 1 5 10 15 20; do
    RESPREFIX="cipsel2_$CIPK"
    STRING=" --cipselector_option 2 --cipselector_k $CIPK --resprefix $RESPREFIX"
#    pass
done

## Normalization:
for NORM in 0; do
    RESPREFIX="no_norm"
    STRING=" --use_normalization $NORM --resprefix $RESPREFIX"
#    pass
done

## Pareto Options: ###### REMOVED 'all' FOR CHEMSETS
for PARETO in 'random' 'greedy' 'paretogreed' 'pareto_only' 'all' 'default'; do
    RESPREFIX="res_pareto_$PARETO"
#    STRING=" --pareto_option $PARETO --resprefix $RESPREFIX"
    STRING=" --core_sizes 0 1 2 3 --pareto_option $PARETO --resprefix $RESPREFIX"
#    reportchem
done

## Contextsizes/Thickness: 
for CONTEXTSIZE in 1 2; do
    RESPREFIX="contextsize_$CONTEXTSIZE"
    STRING=" --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    pass
done

## Mincount/min_cip:
for MINCOUNT in 1 2; do
    RESPREFIX="mincount_$MINCOUNT"
    STRING=" --min_count $MINCOUNT --resprefix $RESPREFIX"
#    pass
done

## Graphsizelimiter:
for SIZELIMITER in 0 1; do
    RESPREFIX="sizelimiter_$SIZELIMITER"
    STRING=" --graph_size_limiter $SIZELIMITER --resprefix $RESPREFIX"
#    pass
done


## Cipselector Comparison Graphs
KEEP=60
#CIP0
###for CIPK in 300 1500 3000 4500 6000 7500 9000; do # KEEP 30
###for CIPK in 100 900 2250 3750 5250 6750 8250; do
###for CIPK in 150 750 1500 2250 3000 3750 4500; do # KEEP 60
for CIPK in 50 450 1125 1875 2625 3375 4125; do  
    RESPREFIX="res_keep-$KEEP-cipsel0-$CIPK"
    STRING=" --keepgraphs $KEEP --cipselector_option 0 --cipselector_k $CIPK --pareto_option 'greedy' --resprefix $RESPREFIX"
    report
done
#CIP1
###for CIPK in 10 50 100 150 200 250 300; do # KEEP 30
###for CIPK in 50 30 75 125 175 225 275; do
###for CIPK in 5 25 50 75 100 125 150; do # KEEP 60
for CIPK in 1 15 38 63 88 113 138; do
    RESPREFIX="res_keep-$KEEP-cipsel1-$CIPK"
    STRING=" --keepgraphs $KEEP --cipselector_option 1 --cipselector_k $CIPK --pareto_option 'greedy' --resprefix $RESPREFIX"
    report
done
#CIP2
###for CIPK in 10 20 30 40 50 60 70 80 90 100; do # KEEP 30
###for CIPK in 1 2 3 4 5 6 7 8 9; do
###for CIPK in 5 10 15 20 25 30 35 40 45 50; do # KEEP 60
for CIPK in 1 2 3 4 6 7 8 9; do
    RESPREFIX="res_keep-$KEEP-cipsel2-$CIPK"
    STRING=" --keepgraphs $KEEP --cipselector_option 2 --cipselector_k $CIPK --pareto_option 'greedy' --resprefix $RESPREFIX"
    report
done
