def graphify(image):
    node_id = 0
    T = nx.Graph()
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            node_id  = i*image.shape[1] + j

            # add neighbor nodes
            if (image[i,j]):
                for ii in range(i-1,i+2):
                    for jj in range(j-1,j+2):
                        if ii != i or jj != i:
                            if (image[ii,jj]):
                                nbr_node_id = ii*image.shape[1] + jj
                                T.add_edge(node_id, nbr_node_id)
    return T


def partition(G, orig_size, am_nodes = [], threshold = 0.055):
    T = G.copy()
    print "ambig_nodes orig", len(am_nodes)
    while True:
        if len(am_nodes) > threshold*orig_size:
            print "Too much: ", len(am_nodes), orig_size
            return G, []
        if not nx.is_connected(T) :
            break
        flows = nx.approximate_current_flow_betweenness_centrality(T)

        max_node = max(flows, key=flows.get)
        max_neighbs = T.neighbors(max_node)
        T.remove_node(max_node)
        am_nodes.append((max_node, max_neighbs))
    ccs = nx.connected_components(T)

    print "Let's gooo: ", len(am_nodes), orig_size
    return T, am_nodes

def build_partition(T_part, image, ambig_nodes =[]):
    ccs = nx.connected_components(T_part)
    ccs = list(enumerate(list(ccs)))
    #ccs = T_part
    ambig_nodes = dict(ambig_nodes)
    print ambig_nodes
    #ambig_color = 1 if (len(ccs[0]) > len(ccs[1])) else 2

    image2 = np.zeros((image.shape[0],image.shape[1]))
    for i in range(1,image.shape[0]-1):
        for j in range(1,image.shape[1]-1):
            node_id  = i*image.shape[1] + j
            if node_id in ambig_nodes.keys():
            	for item in ccs:
            		if ambig_nodes[node_id][0] in item[1]:
            			image2[i,j] = item[0] + 1
                
            else:
		        for item in ccs:
		            #print ccs[k]
		            if node_id in item[1]:
		            	image2[i,j] = item[0] + 1
                
    return image2

def community_segmentation(image):
	T = graphify(image)
	ccs = list(nx.connected_components(T))
	T_fin = nx.Graph()
	print "Total len: ", len(T)
	total_amb_nodes = []
	for comp in ccs:
		k = T.subgraph(comp) 
		T_part, ambig_nodes = partition(k, len(k), am_nodes = [])
		total_amb_nodes += ambig_nodes
		T_fin = nx.compose(T_fin,T_part)
	image2 = build_partition(T_fin, image, ambig_nodes = total_amb_nodes)
	return image2