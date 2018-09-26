





import graphlearn3.lsgg as lsgg
def rule_rand_graphs(input_set, numgr =100, iter= 1):

    grammar = lsgg.lsgg(decomposition_args={"radius_list": [0, 1],
                                     "thickness_list": [ 2],
                                     'hash_bitmask': lsgg._hash_bitmask_},
                 filter_args={"min_cip_count": 1,
                              "min_interface_count": 2},
                 cip_root_all = False,
                 half_step_distance= False )
    grammar.fit(input_set)

    import graphlearn3.util as u
    u.draw_grammar_term(grammar)
    for i in range(iter):
        input_set = [g for start in input_set for g  in grammar.neighbors(start)][:numgr]


    # also needs duplicate removal

    return input_set













def test_rulerand():
    import util.random_graphs as rg
    import structout as so
    grs  = rg.make_graphs_static()[:10] # default returns 100 graphs..
    res =rule_rand_graphs(grs, numgr = 100)
    res =rule_rand_graphs(res[:10], numgr = 100)
    so.gprint(res)