import networkit as nk
import numpy as np
from sklearn.datasets import make_blobs

def create_opinions(pop_size,opinion_dims,num_clusters,random_seed=None):
    """
    generate random opinions for each member of population
    
    plt.hist(x) to see distribution for each dimension
    """
    # opinion centres for each cluster
    centers = [[c]*opinion_dims for c in np.arange(0,1,1/(num_clusters+1))[1:]]
    # make clustered data
    x, labels_true = make_blobs(
        n_samples=pop_size, n_features=opinion_dims, centers=centers, cluster_std=[0.05]*num_clusters, random_state=random_seed
    )
    # return opinion vector/matrix
    return x

def create_graph(pop_size,graph_type='complete'):
    """
    generate a graph for population size
    """
    # complete graph
    if graph_type == 'complete':
        G = nk.generators.ErdosRenyiGenerator(pop_size, 1).generate()
    
    # social graph
    if graph_type == 'social':
        # Initalize algorithm
        lfr = nk.generators.LFRGenerator(n=pop_size) # num. nodes
        # Generate sequences
        lfr.generatePowerlawDegreeSequence(pop_size/min(pop_size/4,100), pop_size/10, -2) # avg. degree, max. degree, exp. of degree distr.
        lfr.generatePowerlawCommunitySizeSequence(pop_size/min(pop_size/4,100), pop_size/2, -1) # min, max, exp.
        lfr.setMu(0.5) # mixing parameter
        # Run algorithm
        uG = lfr.generate()
        G = nk.Graph(uG, directed=True)
    
    return G