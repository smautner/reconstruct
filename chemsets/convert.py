
import eden_chem.io.rdkitutils as ru
import pickle 
from networkx.readwrite import json_graph
import json

def convert(AID = 'AID828.sdf'):
    nxgr = [ json_graph.node_link_data(g) for g in  ru.sdf_to_nx(AID)]
    #for f in nxgr: f.graph={}
    with open(AID+".json", 'w') as handle:
        handle.write(json.dumps(nxgr))



if __name__=="__main__":
    import sys
    convert(sys.argv[1])

