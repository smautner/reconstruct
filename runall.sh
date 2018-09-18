rm ".res/"*
rm -r ".log/1"
echo -e (seq 0 0)" "(seq 0 3)" "(seq 0 3)" "(seq 0 9)"\n" | parallel --bar --results .log -j 15 python3 reconstruct.py
