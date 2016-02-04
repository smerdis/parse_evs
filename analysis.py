# copy the membership stuff from plot_corrs_arjun

# g = igraph object = matrix_to_igraph(matrix, cost=.1)
# take the existing comm memberships, calculate modularity

g = brain_graphs.brain_graph(VertexClustering(g, membership=memb))
# to make own clustering, g.infomap or g.clustering or w/e
q = g.community.modularity
g.pc, g.wmd
