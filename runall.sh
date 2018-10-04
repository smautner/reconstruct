rm ".res/"*
rm -r ".log/1"
echo -e (seq 0 11)" "(seq 0 0)" "(seq 0 0)" "(seq 0 19)"\n" | parallel --bar --results .log -j 27 python3 reconstruct.py
