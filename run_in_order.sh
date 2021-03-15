#!/bin/bash

source /beegfs/work/workspace/ws/fr_mh595-conda-0/conda/etc/profile.d/conda.sh
conda activate binenv

REPEATS=50 ### Change to 100 for Normal or 250 for Chem

execute () {    #### Make sure to change filename in first sed command to 'chem_runall_binac.sh' or just 'runall_binac.sh'
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
                            report ##  Replace this with report/execute/pass
                        done
                    done
                done
            done
        done
    done
done
## Chemset Comparison
#for CONTEXTSIZE in 1 2; do
#    for CIPK in 200 300 400; do
#        RESPREFIX="cipK-$CIPK-contextsize-$CONTEXTSIZE"
#        STRING=" --context_size $CONTEXTSIZE --cipselector_k $CIPK --resprefix $RESPREFIX"
#        pass
#    done
#done

## Artificial Comparison
#for CONTEXTSIZE in 1 2; do
#    RESPREFIX="coresizes-012-contextsize-$CONTEXTSIZE"
#    STRING=" --core_sizes 0 1 2 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
#    RESPREFIX="coresizes-01-contextsize-$CONTEXTSIZE"
#    STRING=" --core_sizes 0 1 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
#    RESPREFIX="coresizes-0-contextsize-$CONTEXTSIZE"
#    STRING=" --core_sizes 0 --context_size $CONTEXTSIZE --resprefix $RESPREFIX"
#    report
#done


## Cipselector 1: 100 200 400 800
#for CIPK in 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200; do
for CIPK in 10 30 50 70 90; do
    RESPREFIX="cipsel1_$CIPK"
    STRING=" --cipselector_option 1 --cipselector_k $CIPK --resprefix $RESPREFIX"
#    pass
done

## Cipselector 2: (Default) 1 5 10 15 20 #### REMOVED 100 FOR CHEMSETS
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
for PARETO in 'random' 'greedy' 'pareto_only' 'all' 'default'; do
    RESPREFIX="pareto_$PARETO"
    STRING=" --pareto_option $PARETO --resprefix $RESPREFIX"
#    pass
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
