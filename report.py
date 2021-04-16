import os
from collections import defaultdict
import numpy as np
import pandas
from maketasks import EXPERIMENT_REPEATS, params_graphs, tasklist
from reconstruct2 import instancemakerparams, Optimizerparams, params_opt, tasklist, maketasks
from util.util import jloadfile




'''
i should just be able to use loadblock... 
'''








elif sys.argv[1] == "report":
report(resprefix, tasklist)
elif sys.argv[1] == "reportchem":
tasklist = get_chem_filenames()
report(resprefix, tasklist)
exit()


def getvalue(p, nores, nosucc, folder): # nosucc and nores are just collecting stats
    completed = 0
    allsteps=[-1]
    success = 0
    times = []
    average_productions = []
    for task in range(EXPERIMENT_REPEATS):
        taskname = "%d" % (p+task)
        fname = folder+"/"+taskname
        if os.path.isfile(fname):
            completed +=1
            res, steps, time, avg_productions = jloadfile(fname)
            times.append(time)
            average_productions.append(avg_productions)
            success += res
            if not res:   # FAIL
                nosucc.append(taskname)
            else:       # success -> remember step count
                allsteps.append(steps)
        else:
            nores.append(taskname)
    allsteps = np.array(allsteps)
    times = np.array(times)
    average_productions = np.array(average_productions)
    return success,  allsteps.max(), times.mean(), average_productions.mean()


def report(folder = '.res', tasklist=None):

    problems = id_to_options(tasklist= tasklist)
    print(len(problems))

    dat= defaultdict(lambda: defaultdict(list))
    nores = []
    nosucc =[]
    for p in range(0, len(problems), EXPERIMENT_REPEATS):
        a,b,c,_ = [ problems[p][k] for k in [0,1,2,3]]
        im = imtostr(b)
        gr = grtostr(a)
        op = optitostr(c)
        y,z = im.split(" ")
        dat[y][z] += [getvalue(p, nores, nosucc, folder)]

    #mod = lambda x : str(x).replace("_",' ')
    lsuccess = [int(succ) for data in dat.values() for v in data.values() for succ,steps,times,avg in v]
    avg_productions = np.array([int(avg) for data in dat.values() for v in data.values() for succ,steps,times,avg in v])
    rnd = [int(steps) for data in dat.values() for v in data.values() for succ,steps,times,avg in v]
    print ("nores",nores)
    print ('nosucc',nosucc)
    print ("sumsuccess:", sum(lsuccess), lsuccess)
    print ("Average productions:", avg_productions.mean(), avg_productions)
#    print ("maxrnd:", max([int(b) for c in dat.values() for a,b,_ in c.values()]))
    print("maxrnd:", max(rnd))

    print (pandas.DataFrame(dat).to_string())
    # Terminal exec(cat all the logs|awk "average productions per graph")
    #print (pandas.DataFrame(dat).to_latex())


def defaultformatter(paramsdict, instance_dict):
    res =[]
    for k in paramsdict['keyorder']:
        if len(paramsdict[k] )> 1:
            #  interesting key
            res.append("%s:%s " % ( k[:4],str(instance_dict[k])) )
    res=  tuple(res) or "lol"
    return res


def imtostr(im):
    d=instancemakerparams[im]
    return "marks:%d neigh:%d" % (d["n_landmarks"], d["n_neighbors"])


def optitostr(op):
    d=Optimizerparams[op]
    return defaultformatter(params_opt,d)


def grtostr(gr):
    d = tasklist[gr]
    return defaultformatter(params_graphs, d)
    #return tuple(("Cyc:%d elab:%d nlab:%d siz:%d dist:%s" % (d['allow_cycles'],d['edge_labels'],d['node_labels'],d['size_of_graphs'],d['labeldistribution'][0])).split(" "))
    #return tuple(("elab:%d nlab:%d" % (d['edge_labels'],d['node_labels'])).split(" "))
    #return tuple(("elab:%d nlab:%d graphs:%d rrg_it:%d" % (d['edge_labels'],d['node_labels'],d['number_of_graphs'],d['rrg_iter'])).split(" "))


def id_to_options(tasklist=tasklist):
    params_args = {"keyorder":[3,2,1,0], # 3 first -> order works out ineval
                        0:range(len(tasklist)),
                        1:range(len(instancemakerparams)),
                        2:range(len(Optimizerparams)),
                        3:range(EXPERIMENT_REPEATS),
                        }
    args = maketasks(params_args)
    return args