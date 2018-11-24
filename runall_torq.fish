



for i in (seq 1 100)
    set word "acid_$i"
    qsub -q short torq_single.sh -F "$i" -N "$word"
end

# had jobs grouped, many small mem jobs are better...
#for i in (seq 0 36)
#    set start (math 1+$i\*20)
#    set end (math $start+19)
#    echo $start $end
#    qsub -q short torq.sh -F "$start $end"
#end
