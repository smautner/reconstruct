import reconstruct as r

graphsets = r.loadfile(".tasks")


from structout import gprint
for graphs in graphsets:
    print ("inspectin dataset")
    for i in range(5):
        gprint(graphs[i*5:i*5+5])

