import random



import graphlearn3.util.util as u
from exploration.pareto  import MYOPTIMIZER as myop



import graphlearn3.lsgg as lsgg
def rule_rand_graphs(input_set, numgr =100, iter= 1):



    # make grammar, fit on input
    grammar = lsgg.lsgg(decomposition_args={"radius_list": [1,2],
                                     "thickness_list": [ 1] },
                 filter_args={"min_cip_count": 1,
                              "min_interface_count": 2},
                 cip_root_all = False,
                 half_step_distance= False )
    grammar.fit(input_set)

    cleaner = myop()
    list(cleaner.duplicate_rm(input_set)) # makes sure that we never ouitput the input. dup_rm saves all hashes

    for i in range(iter):
        input_set = [g for start in input_set for g  in grammar.neighbors(start)]
        input_set = cleaner.duplicate_rm(input_set)
        random.shuffle(input_set)
        input_set= input_set[:numgr]


    # also needs duplicate removal

    return input_set, grammar







def test_rulerand():
    import util.random_graphs as rg
    import structout as so
    import graphlearn3.util.setoperations as setop
    grs  = rg.make_graphs_static()[:30]#[:10] # default returns 100 graphs..
    res1, grammar1 =rule_rand_graphs(grs, numgr = 500,iter=2)
    #res, grammar2=rule_rand_graphs(res1, numgr = 50, iter=1)
    #so.gprint(res) #!!!!!!!!

    '''
    print("initial grammar:")
    print (grammar1)
    print("grammar after iteration")
    print (grammar2)
    inter = setop.intersect(grammar1, grammar2)
    print("intersection between grammars:")
    print (inter)
    '''


    #print("grammar1 - grammar2")
    #diff = setop.difference(grammar1,grammar2)
    #u.draw_grammar_term(diff)


    #u.draw_grammar_term(unique2)  # !!!!!!!!!!!!!!!!!!!!!!!!!
    #print ("generated %d graphs" % len(res1))
