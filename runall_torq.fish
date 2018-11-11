


for i in (seq 0 35)
    set start (math 1+$i\*20)
    set end (math start+19)
    qsub -q short torq.sh -F "$start $end"
end
