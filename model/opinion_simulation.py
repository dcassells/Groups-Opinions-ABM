import numpy as np
import networkit as nk
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.utils import shuffle
from sklearn.neighbors import KernelDensity
from model.group_detection import hdb_scan_group, boundary_min_group, fuzzy_group

###############################################################################
###############################################################################
# Create opinion space, and diagnostics on the space (polarization and graph)
###############################################################################
###############################################################################
class opinion_space(object):
    """
    x: the opinion vector/matrix
    G: graph by which agents are connected (networkit)
    A: the adjacency matrix
    b: the bias vector
    """
    def __init__(self, x, G, b=None, bounded=False):
        
        # opinion vector/matrix - ndim for consistency
        self.x = np.array(x)
        
        # opinion history
        self.opinion_history = [self.x]
        
        # length of data, m, and opinion dimensions, n,
        self.m, self.n = self.x.shape
        
        # graph
        self.G = G
        
        # adjacency matrix from graph
        self.A = nk.algebraic.adjacencyMatrix(self.G, matrixType='sparse')
        
        # graph history, stored when it changes
        self.G_history = [(len(self.opinion_history)-1,nk.Graph(self.G))]
        
        # initiate bias vector
        self.b = b
        
        # bounded, or not
        self.bounded = bounded
    
    ###############################################################################
    # Polarization Diagnostics
    def disagreement_index(self, history=False):
        """
        distribution diagnostic: disagreement index
        """
        # x - x.T is symmetric, so could speed up if need
        return [(A_t*((x_t - x_t.T)**2)).sum() for x_t, A_t in zip(self.opinion_history, self.G_history)]
    
    def std_dev(self):
        """
        distribution diagnostic: standard deviation of opinion dimensions
        """
        return [list(np.std(x_t, axis=0)) for x_t in self.opinion_history]
    
    def kurtosis(self):
        """
        distribution diagnostic: kurtosis of opinion dimensions
        """
        return [list(sp_kurtosis(x_t, axis=0)) for x_t in self.opinion_history]
    
    def der(self, alpha=0.5):
        """
        distribution diagnostic: DER (Duclos, Esteban, Ray)
        """
        def kde_solver(X, kernel='gaussian', bandwidth=0.2):
            # fit the kde - needed for DER
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
            
            return kde
        
        """
        # unused
        def kde_preds(kde, lower_bound=0, upper_bound=1, definition=100):
    
            # predictions to plot the kde
            x_mesh = np.linspace(lower_bound, upper_bound, num=definition).reshape(-1, 1)
            log_dens = kde.score_samples(x_mesh)
            kde_preds = np.exp(log_dens)
            
            return [x_mesh.flatten(),kde_preds]
        """
        
        def der_measure(X, f, alpha=alpha):
            """
            for data X, with probability distribution f.
            f should be from a scikit package that supports .score_samples - gives log likelihood
            parameter alpha can be altered
            """
            # X must be ordered such that x_1 <= x_2 <= ... <= x_n
            X = np.sort(X,axis=None).reshape(-1,1)
            # size of data
            n = len(X)
            # array of all 'i's
            n_range = np.arange(1,n+1).reshape(-1,1)
            # calculate mean
            mu = X.mean()
        
            # we calculate f_hat
            # .score_samples returns log likelihood so need np.exp()
            f_hat = np.exp(f.score_samples(X)).reshape(-1,1)
            
            # calculate a_hat, begin with everything left of j=1 sum then add in that part
            a_hat = np.full((n,1),mu) + X*((2*n_range - 1)/n - 1)
            
            j_sum = X.reshape(-1,1).copy()
            for i in range(1,n):
                j_sum[i] = 2*(X[:i].sum()) + j_sum[i]
            
            a_hat = a_hat - j_sum/n
            
            # put the parts of the formula together
            der = ((f_hat**alpha)*a_hat).sum()/n
            
            return der
        
        der_history = []
        for x_t in self.opinion_history:
            der_x_t = []
            for dim in range(self.n):
                X = x_t[:,dim].reshape(-1, 1)
                estimated_distribution = kde_solver(X)
                der_iter = der_measure(X, estimated_distribution)
                der_x_t.append(der_iter)
            der_history.append(der_x_t)
        
        return der_history
    
    ###############################################################################
    # Graph Diagnostics
    def node_opinion_sum(self):
        """
        graph diagnostic: sum of neighbours' opinions connected to node i
        """
        # dot adj matrix with opinion matrix
        return self.A.dot(self.x)
    
    def degree_distribution(self, G=None):
        """
        graph diagnostic: returns degree of each node
        """
        if G is None:
            G=self.G
        return sorted(nk.centrality.DegreeCentrality(G).run().scores())
    
    def connected_components(self, G=None):
        """
        graph diagnostic: return number of connected components, with possibility to return labels too
        """
        if G is None:
            G=self.G
        return nk.components.ConnectedComponents(G).run().numberOfComponents()
    
    def local_clustering_coefficient(self, G=None):
        """
        graph diagnostic: return the local clustering coefficient
        """
        if G is None:
            G=self.G
        return nk.centrality.LocalClusteringCoefficient(G).run().ranking()
    
    def global_clustering_coefficient(self, G=None, exact=True):
        """
        graph diagnostic: return the global clustering coefficient, can use approx algorithm
        """
        if G is None:
            G=self.G
        if exact:
            return nk.globals.ClusteringCoefficient.exactGlobal(G)
        else:
            return nk.globals.ClusteringCoefficient.approxGlobal(G, trials=10000)
    
    def modularity(self):
        """
        graph diagnostic: return modularity of matrix
        """
        return None
        

###############################################################################
###############################################################################
# Create simulation space; inherits the opinion space and provides rules to change it
###############################################################################
###############################################################################
class platform_simulation(opinion_space):
    """
    expected_posts: the expected number of posts an actor makes during each iteration
    p_read: the probability that an actor reads their feed
    opinion_rule: which opinion rule to default to
    rewire_rule: which rewiring rule to default to
    feed_ordering: decides how feed is ordered; if left as None then it is a random shuffle
    """
    def __init__(self, x, G, b=None, bounded=False, expected_posts=None, p_read=None, opinion_rule=None, rewire_rule=None, group_method=None,
                 feed_ordering=None, ideological_limit=1, edge_creation_limit=0.1,
                 R_in=0.25, R_out=0.25, T_in=0.5, T_out=0.5, E_in=0.1, E_out=0.1):
        
        # complete initialisation from opinion_space
        super().__init__(x, G, b, bounded)
        
        # post probabilities - if not provided, set to 1 (i.e. always post)
        self.expected_posts = expected_posts
        if self.expected_posts is None:
            # expected num. posts for each actor drawn from exponential distr.
            # parameter can be tuned, or expected posts empirically inferred
            exponential_dist_param = 0.75
            self.expected_posts = np.random.exponential(exponential_dist_param, self.m)
        
        # indicating if node i posted - 0/1
        self.actual_posts = None
        
        # read probabilities - if not provided, set relative to opinion distance
        # this will also be influenced by recommendation algorithms at some point
        self.p_read = p_read
        
        # feed for information posted
        self.feed = {}
        
        # indicating what nodes i read - indices
        self.x_read = {}
        
        # ideological distances too far so remove edge
        self.x_remove_edges = []
        
        # ideological limit distance for removing edges
        self.ideological_limit = ideological_limit
        
        # ideological distance below which new edges are created (if they don't already exist)
        self.edge_creation_limit = edge_creation_limit
        
        # R/T/E for opinion_attraction_repulsion
        self.R_in = R_in
        self.R_out = R_out
        self.T_in = T_in
        self.T_out = T_out
        self.E_in = E_in
        self.E_out = E_out
        
        # group information
        self.group_method = group_method
        self.group_detection(method=self.group_method)
        if self.group_method == 'fuzzy':
            self.group_history = [self.group_matrix]
        else:
            self.group_history = [self.group_id]
        
        # rule for updating opinions
        self.opinion_rule = opinion_rule
        
        # rule for rewiring network, if any
        self.rewire_rule = rewire_rule
        
        # feed ordering rule
        self.feed_ordering = feed_ordering
        
    ###############################################################################
    # Information Gathering Processes
    def feed_populate(self):
        """
        info exchange: method to determine whether or not node i posts information
        """
        # actual posts drawn from Poisson distribution
        self.actual_posts = np.random.poisson(lam=self.expected_posts, size=(self.m,))
    
    def feed_order(self):
        """
        info exchange: create the feed for each user
        """
        # reinitialise feed as empty
        self.feed = {}
        # populate feed
        for i in range(self.m):
            # create mask for where subscriptions exist and nodes have posted
            mask = (self.A[i].toarray()==1).flatten() & (self.actual_posts > 0)
            # available feed populated from the mask
            avail_feed = list(zip(np.argwhere(mask).flatten(), self.actual_posts[mask]))
            flat_feed = [item for x in avail_feed for item in [x[0]]*x[1]]
            
            
            # order the feed
            # shuffle the order of the feed
            if self.feed_ordering is None:
                self.feed[i] = shuffle(flat_feed)
            # most frequent posters to the top
            elif self.feed_ordering == 'loudest':
                avail_feed.sort(key=lambda x: -x[1])
                self.feed[i] = [item for x in avail_feed for item in [x[0]]*x[1]]
            # recommender system as an option
    
    def feed_read(self):
        """
        info exchange: governs how the feed is read - proportional to position on feed (wrt to length) with exp decay
                        each user has some constant multiplier representing propensity to read the feed on top of this
        """
        # for the moment we assume that node i reads the top 5 posts of their feed
        self.x_read = {}
        self.x_remove_edges = set()
        # read from feed
        for i in range(self.m):
            self.x_read[i] = self.feed[i][:5]
            # if ideological distances too far then remove edge
            for j in self.x_read[i]:
                if np.linalg.norm(self.x[i] - self.x[j]) > self.ideological_limit:
                    if self.G.isDirected():
                        self.x_remove_edges.add((i, j))
                    elif (j, i) not in self.x_remove_edges:
                        self.x_remove_edges.add((i, j))
    
    # generate exchanges - wraps the other functions in the information gathering section
    def info_gather(self):
        """
        info exchange: run the steps of the information gathering process in one function
        """
        self.feed_populate()
        self.feed_order()
        self.feed_read()
        
    ###############################################################################
    # Influence Network
    #### sometimes we want to circumvent the feed populating and ordering process
    #### instead here we just ready opinion update by mirroring the network structure
    def influence_network(self,E_in=None,E_out=None):
        """
        this circumvents the feed creation/order/read steps (info_gather())
        """
        self.x_read = {}
        self.x_remove_edges = set()
        # convert to list of lists format and return nonzero indices for each row (where there exists an edge)
        edge_list = self.A.tolil().rows
        # seed for iteration - set it to what num iteration on (equals length of opinion history)
        random_seed = len(self.opinion_history)
        # maps edges to 'x_read' and flag edges to remove too
        for i in range(len(edge_list)):
            neighbours = set(edge_list[i])
            in_group = set((self.group_matrix[i]==1).nonzero()[0])
            out_neighbours = neighbours - in_group
            in_neighbours = list(neighbours - out_neighbours)
            out_neighbours = list(out_neighbours)

            def probability_calc(E,neighbours):
                # calculcate ideological distances 
                distances = np.linalg.norm(self.x[i] - self.x[neighbours],axis=1)
                
                # probabilities of speaking
                probabilities = (0.5)**abs(distances/E)
                
                # which actors to update
                np.random.seed(random_seed)
                update_actors = np.random.random((1,len(neighbours))) <= probabilities
                
                return np.array(neighbours).reshape(1,len(neighbours))[update_actors].tolist()

            in_neighbours = probability_calc(E_in, in_neighbours)
            out_neighbours = probability_calc(E_out, out_neighbours)
            
            self.x_read[i] = in_neighbours + out_neighbours
            
            # if ideological distances too far then remove edge, setting large ideological_limit can avoid this
            for j in self.x_read[i]:
                if np.linalg.norm(self.x[i] - self.x[j]) > self.ideological_limit:
                    if self.G.isDirected():
                        self.x_remove_edges.add((i, j))
                    elif (j, i) not in self.x_remove_edges:
                        self.x_remove_edges.add((i, j))
    
    ###############################################################################
    # Group Detection
    #### to implement group dynamics on top of one-to-one interactions
    def group_detection(self,method):
        if method == 'hdb':
            self.group_matrix, self.group_id = hdb_scan_group(self.x)
        elif method == 'error_min':
            self.group_matrix, self.group_id, _ = boundary_min_group(self.x)
        elif method == 'fuzzy':
            self.group_matrix, _ = fuzzy_group(self.x) # should provide pdf here, otherwise it's calculated twice
            self.group_id = np.full((self.m),99)
        else:
            raise Exception('Group identity method not defined')
    
    ###############################################################################
    # Opinion Updating Processes
    #### agent opinion updating rules
    def is_bounded(self):
        # if space is bounded then clip opinion vector/matrix to 0,1
        if self.bounded:
            self.x = np.clip(self.x,0,1)
    
    def opinion_averaging(self):
        """
        opinion update: deGroot - take the average opinion of what has been read
        """
        # w_ii is weight of own opinion
        w_ii = 1
        # matrix to easily sum read neighbour opinions
        update_matrix = np.zeros((self.m, self.m))
        for i in range(self.m):
            for j in self.x_read[i]:
                update_matrix[i, j] += 1
        # s if the sum of read opinions
        s = update_matrix.dot(self.x)
        # d is the number of opinions read
        d = update_matrix.sum(axis=1,keepdims=True)
        # update opinion matrix
        self.x = (w_ii*self.x + s)/(w_ii + d)
        # if bounded then results are clipped
        self.is_bounded()
        # add to history
        self.opinion_history.append(self.x)
        # update opinion clusters, if needed
        self.group_detection(method=self.group_method)
        if self.group_method == 'fuzzy':
            self.group_history.append(self.group_matrix)
        else:
            self.group_history.append(self.group_id)
        
    def opinion_attraction_repulsion(self):
        """
        opinion update: Axelrod 2021    - if opinion distance below a threshold then attract, else repulse
                                        - E is an interaction probability parameter from the original model used in interaction probability
        """
        # cumulative position update to change x at the end of each iteration, rather than each interaction
        position_updates = np.zeros((self.m,self.n))
        # fill position updates
        for i in self.x_read.items():
            for j in i[1]:
                # should multi-D ideological distance be element-wise or euclidean(/other?) norm?
                opinions_diff = self.x[j] - self.x[i[0]]
                distance = np.linalg.norm(opinions_diff)
                # update ideology amount
                # if same cluster
                if self.group_matrix[i[0],j] == 1:
                    if distance <= self.T_in:
                        position_updates[i[0]] += +self.R_in*opinions_diff
                    else:
                        position_updates[i[0]] += -self.R_in*opinions_diff
                # else out group
                else:
                    if distance <= self.T_out:
                        position_updates[i[0]] += +self.R_out*opinions_diff
                    else:
                        position_updates[i[0]] += -self.R_out*opinions_diff
        
        # update opinion matrix
        self.x = self.x + position_updates
        # if bounded then results are clipped
        self.is_bounded()
        # add to history
        self.opinion_history.append(self.x)
        # update opinion clusters
        self.group_detection(method=self.group_method)
        if self.group_method == 'fuzzy':
            self.group_history.append(self.group_matrix)
        else:
            self.group_history.append(self.group_id)


    ###############################################################################
    # Network Updating Processes
    #### possibility that when ideological distances between neighbours are too big, nodes rewire themselves in the network
    
    # REMOVING EDGES
    def remove_edges(self):
        """
        network update: remove edges
        """
        # bool to update graph history/A
        if len(self.x_remove_edges) > 0:
            update_graph = True
        else:
            update_graph = False
        
        # new_subscription_nodes = []
        while len(self.x_remove_edges) > 0:
            edge = self.x_remove_edges.pop()
            self.G.removeEdge(edge[0],edge[1])
        
        # if x_remove_edges not empty then add to graph history, and update A
        if update_graph:
            # graph history
            new_graph = nk.Graph(self.G)
            self.G_history.append((len(self.opinion_history)-1,new_graph))
            # adjacency matrix
            self.A = nk.algebraic.adjacencyMatrix(self.G, matrixType='sparse')

    # ADDING EDGES
    def new_edges(self):
        """
        network update: create new edges
        """
        # if ideologically close nodes and no edge, then connect
        node_set = set(np.arange(self.m))
        edge_list = self.A.tolil().rows
        # iterate through nodes
        for i in range(self.m):
            # list of unsubscribed to neighbours
            unsubbed_neighbours = list(node_set - set(edge_list[i]) - set([i]))
            for j in unsubbed_neighbours:
                if np.linalg.norm(self.x[i] - self.x[j]) < self.edge_creation_limit:
                    self.G.addEdge(i,j)
                
    
    # wraps the network updating process
    def network_update(self):
        """
        network update: runs steps of network updating procedure
        """
        self.new_edges()
        self.remove_edges()

    ###############################################################################
    # Run Info Gather, Opinion Update, Network Update Process
    def run_iteration(self, feed_mute = None):
        # update opinions via the feed, or via influence network
        if feed_mute is None:
            self.info_gather()
        elif feed_mute == 'influence':
            self.influence_network(self.E_in,self.E_out)
        else:
            raise ValueError('unknown feed_mute option given')
        # opinion updating
        self.opinion_attraction_repulsion()
        # network updating
        self.network_update()