"""
Methods to identify groups on an opinion distribution:
    1) HDBScan - clustering algorithm
    2) Boundary Error Minimisation - implementation of method from Yang (2018)
    3) Fuzzy Clustering - an 'ego-centric'

Input - an array of values representing an opinion distribution (this can be multidimensional)

Output - an n x n array such that if i considers j as 'in-group' then output_{i,j} = 1, else 0
"""
####################
##### imports and helpful functions
import numpy as np
from sklearn.cluster import HDBSCAN
import model.hdb_fuzzy_clustering as hdb_fuzz
import scipy.integrate as integrate
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def id_to_matrix(group_id):
    """
    for global methods (all actors agree on group identities), a vector of labels is turned into a matrix
    for local methods, this is done in its own function
    """
    
    num_actors = group_id.shape[0]
    # initialise as identity matrix (actors are in the same group as themselves)
    group_matrix = np.identity(num_actors)
    
    for i in range(num_actors-1):
        # returns 1 where group identities of i and j match, else 0
        # membership is symmetric when this function is used
        # we don't need the final actor since relationship with all others has already been calculated
        sub_array = np.where(group_id[i+1:]==group_id[i], 1, 0)
        group_matrix[i, i+1:] = sub_array
        group_matrix[i+1:, i] = sub_array

    return group_matrix

def fit_hdb_scan(x):
    # using sklearn HDBSCAN
    max_clusters = 5
    hdb = HDBSCAN(min_cluster_size=max(2,int(x.shape[0]/max_clusters)),allow_single_cluster=True)
    # fit algo
    return hdb.fit(x)

####################
##### 1) HDBScan
def hdb_scan_group(x):
    clusterer = fit_hdb_scan(x)
    group_id = clusterer.labels_
    
    return id_to_matrix(group_id), group_id

####################
##### 2) Boundary Error Minimisation
def boundary_min_group(x,pdf=None,a=0,b=1,num_groups=2,x0=0.5):
    """
    x : opinion distribution
    pdf : underlying pdf of x, if known
    a : lower bound of opinion space
    b : upper bound of opinion space
    num_groups : number of groups to define
    x0 :  initial guess at boundary position
    """
    
    def group_mean(func,l_bound,u_bound):
        """
        we need to calculate a mean position of the group, following equation (1)
        
        func    : density function of the opinion distribution 
        l_bound : lower bound of the space
        u_bound : upper bound of the space
        """
        # return value of integrate.quad is a tuple, with the first element holding the estimated value
        # of the integral and the second element holding an upper bound on the error
        return (integrate.quad(lambda x: x*func(x), l_bound, u_bound)[0]
                /integrate.quad(lambda x: func(x), l_bound, u_bound)[0])

    def individual_error(u,z,func,l_bound,u_bound,group):
        """
        calculate individual error of individual u for boundary position z, following equation (3)
        
        u       : the position of individual u
        z       : the position of the group boundary
        func    : density function of the opinion distribution
        l_bound : lower bound of the space
        u_bound : upper bound of the space
        group   : which group are we looking for the boundary of
        
        group 1 < group 2
        """
        
        # difference between group means given the boundary
        group_mean_difference = abs(group_mean(func,z,u_bound) - group_mean(func,l_bound,z))
        
        # limits on the integrals depend on what group we are looking for
        if group == 0:
            bound_1, bound_2, bound_3 = l_bound, z, u_bound
        elif group == 1:
            bound_1, bound_2, bound_3 = u_bound, z, l_bound
        
        # in group error
        in_group = integrate.quad(lambda v: abs(u-v)**2 * func(v), bound_1, bound_2)[0]
        # out group error
        out_group = integrate.quad(lambda v: (group_mean_difference - abs(u-v))**2 * func(v), bound_2, bound_3)[0]
        
        # correct for the boundary order if group 2 by flipping the sign
        if group == 1:
            in_group, out_group = -in_group, -out_group
        
        return in_group + out_group

    def global_error(z,func,l_bound,u_bound,group):
        """
        calculate global error if boundary position is at z, following equation (4)
        
        u       : the position of individual u
        z       : the position of the group boundary
        func    : density function of the opinion distribution
        l_bound : lower bound of the space
        u_bound : upper bound of the space
        group   : which group are we looking for the boundary of
        
        group 1 < group 2
        """
        
        # limits on the integrals depend on what group we are looking for
        if group == 0:
            bound_1, bound_2 = l_bound, z
        elif group == 1:
            bound_1, bound_2 = z, u_bound
        
        # normalise with amount of mass within the boundary
        mass_norm = integrate.quad(lambda x: func(x), bound_1, bound_2)[0]
        
        # average individual error
        cat_error = integrate.quad(lambda u: individual_error(u,z,func,l_bound,u_bound,group)*func(u), bound_1, bound_2)[0]
        
        return cat_error/mass_norm

    def kde_solver(X, kernel='gaussian', bandwidth=0.2):
            # fit the kde - needed for DER
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
            
            return kde

    # estimate the pdf if it's not defined
    if pdf is None:
        kde = kde_solver(x)
        # return score samples
        pdf = lambda s: np.exp(kde.score_samples(np.array(s).reshape(-1,1)))#.reshape(-1,1)

    # estimate the group boundaries
    boundaries = dict()
    group_id = np.full((x.shape[0],1),-1)
    for group in range(num_groups):
        # optimization to find min global error boundary position
        res = minimize(global_error, x0, method='nelder-mead', args=(pdf, a, b, group), options={'xatol': 1e-8, 'disp': False})
        # store boundary position
        static_boundary = 0 if group == 0 else 1
        boundaries[group] = np.sort(np.append(res.x,static_boundary)) # sort is needed for next step
        # update group_id
        group_id = np.where((boundaries[group][0] <= x) & (x <= boundaries[group][1]), group, group_id)

    group_id = group_id.flatten()
    
    return id_to_matrix(group_id), group_id, boundaries
    
####################
##### 3) Fuzzy Clustering
def fuzzy_group(x,num_groups=2,random_seed=None):
    """
    x : opinion distribution
    pdf : underlying pdf of x, if known
    a : lower bound of opinion space
    b : upper bound of opinion space
    num_groups : number of groups to define
    x0 :  initial guess at boundary position
    """

    def fuzzy_similarity(i,soft_membership,x,opinion_considered=False,norm_order=np.inf):
        """
        L-inf (supremum) norm between cluster identity array points
        """
        distance_measure = lambda a: np.linalg.norm(a, axis=1, ord=norm_order)
        # soft membership difference
        soft_diff = distance_measure(soft_membership[i]-soft_membership)
        
        if opinion_considered:
            # opinion difference
            op_diff = distance_measure(x[i]-x)
            similarity = (1 - soft_diff)*(1 - op_diff)
        else:
            similarity = (1 - soft_diff)
        
        return similarity

    def strictly_decreasing(vals,range):
        ceiling = 1
        for j in range:
            if vals[j] < ceiling:
                ceiling = vals[j]
            else:
                vals[j] = ceiling

    soft_membership = hdb_fuzz.hdb_fuzzy_clustering(x)
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # initialise group_matrix
    num_actors = x.shape[0]
    group_matrix = np.identity(num_actors)
    
    for i in range(num_actors):
        score = fuzzy_similarity(i,soft_membership,x)
        # make score strictly decreasing
        below = reversed(range(i))
        above = range(i+1,len(score))
        strictly_decreasing(score,below)
        strictly_decreasing(score,above)
        # probability evaluation
        group_matrix[i] = np.random.random((num_actors)) <= score

    return group_matrix, soft_membership