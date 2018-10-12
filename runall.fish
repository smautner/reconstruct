
rm ".res/"*
rm -r ".log/1"
echo -e (seq 0 15)" "(seq 0 0)" "(seq 0 0)" "(seq 0 19)"\n" | parallel --bar --results .log -j 36 python3 reconstruct.py
