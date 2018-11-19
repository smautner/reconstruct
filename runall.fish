


#rm ".res/"*
#rm -r ".log/1"
# x -1 0 makes more sense because higher numbers tend to be harder
echo -e (seq 7 -1 0)" "(seq 0 0)" "(seq 0 4)" "(seq 0 19)"\n" | parallel --bar --results .log -j 36 python3 reconstruct.py
