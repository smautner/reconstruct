rm ".res/"*
rm -r ".log/1"
echo -e (seq 0 11)" "(seq 0 1)" "(seq 0 1)" "(seq 0 9)"\n" | parallel --bar --results .log -j 36 python3 reconstruct.py
