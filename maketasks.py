import random
from reconstruct2 import maketasks
from util import random_graphs as rg, rule_rand_graphs as rrg
from util.util import dumpfile

'''
ok so 
'''

def make_task_file():
    import extensions.lsggscramble  as scram
    data = scram.funmap(maketsk, tasklist,poolsize=20)
    dumpfile(data, ".tasks")
    #dumpfile([ rg.make_graphs_static(maxdeg=3, **args) for args in tasklist], ".tasks")


def maketsk(args):
    rrg_iter = args.pop("rrg_iter")
    graphs = rg.make_graphs_static(**args)
    if rrg_iter > 0:
        graphs = rrg.rule_rand_graphs(graphs, numgr=500+EXPERIMENT_REPEATS,iter=rrg_iter)[0]
    return graphs


def load_chem(AID):
    import json
    import networkx.readwrite.json_graph as sg
    import networkx as nx
    import exploration.pareto as pp
    from structout import gprint
    with open(AID, 'r') as handle:
        js = json.load(handle)
        res = [sg.node_link_graph(jsg) for jsg in js]
        res = [g for g in res if len(g)> 2]
        res = [g for g in res if nx.is_connected(g)]  # rm not connected crap
        for g in res:g.graph={}
        zz=pp.MYOPTIMIZER()
        res2 = list(zz._duplicate_rm(res))
        print ("duplicates in chem files:%d"% (len(res)-len(res2)))
        print (zz.collisionlist)
        #for a,b in zz.collisionlist:
        #    gprint([res[a],res[b]])
        zomg = [(len(g),g) for g in res]
        zomg.sort(key=lambda x:x[0])
        cut = int(len(res)*.1)
        res2 = [b for l,b in zomg[cut:-cut]]
    return res2


def get_chem_filenames():
    # these are size ~500
    files="""AID1224837.sdf.json  AID1454.sdf.json  AID1987.sdf.json  AID618.sdf.json     AID731.sdf.json     AID743218.sdf.json  AID904.sdf.json AID1224840.sdf.json  AID1554.sdf.json  AID2073.sdf.json  AID720709.sdf.json  AID743202.sdf.json  AID828.sdf.json"""
    # these are size ~4000
    files='''AID119.sdf.json
            AID1345082.sdf.json
            AID588590.sdf.json
            AID624202.sdf.json
            AID977611.sdf.json'''
    files = files.split()
    return files


def make_chem_task_file():
    files = get_chem_filenames()
    res=[]
    for f in files:
        stuff =load_chem("chemsets/"+f)
        random.shuffle(stuff)
        res.append(stuff)
    dumpfile(res, ".chemtasks")


EXPERIMENT_REPEATS = 50 #### CHANGE THIS BACK TO 100! 50 only for chemsets
params_graphs = {
    'keyorder' :  ["number_of_graphs", "size_of_graphs","node_labels","edge_labels","allow_cycles","labeldistribution","maxdeg","rrg_iter"],
    'allow_cycles':[True], # cycles are very bad
    'number_of_graphs': [30],
    'size_of_graphs' :[8] ,
    'node_labels' : [4],
    'edge_labels' : [2], # using 5 here mega ga fails
    'labeldistribution': ['uniform'] ,# real is unnecessary
    'maxdeg':[3],
    # rule rand graphs , iter argument ,
    #0 means just use the rand graphs, a little hacky but works for now
    'rrg_iter':[3]
}
tasklist  = maketasks(params_graphs ) # boring task list



if __name__=="__main__":

elif sys.argv[1]=="maketaskschem":
    print("writing task file...")
    make_chem_task_file()
    exit()
