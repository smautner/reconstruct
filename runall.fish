


rm ".res/"*
rm -r ".log/1"
# x -1 0 makes more sense because higher numbers tend to be harder
echo -e (seq 1 450)"\n" | parallel --bar --results .log -j 15 python3 reconstruct.py
