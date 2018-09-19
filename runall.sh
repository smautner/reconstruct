rm ".res/"*
rm -r ".log/1"
echo -e (seq 0 7)" "(seq 0 5)" "(seq 0 0)" "(seq 0 9)"\n" | parallel --bar --results .log -j 36 python3 reconstruct.py
