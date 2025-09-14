import hdbscan
import numpy as np
import pandas as pd

from scipy.spatial.distance import cdist

"""
This is an implementation of the hdbscan fuzzy clustering detailed here:
    https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
"""

"""
Distance based membership
"""
def exemplars(cluster_id, condensed_tree):
    raw_tree = condensed_tree._raw_tree
    # Just the cluster elements of the tree, excluding singleton points
    cluster_tree = raw_tree[raw_tree['child_size'] > 1]
    # Get the leaf cluster nodes under the cluster we are considering
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster_id)
    # Now collect up the last remaining points of each leaf cluster (the heart of the leaf)
    result = np.array([])
    for leaf in leaves:
        max_lambda = raw_tree['lambda_val'][raw_tree['parent'] == leaf].max()
        points = raw_tree['child'][(raw_tree['parent'] == leaf) &
                                   (raw_tree['lambda_val'] == max_lambda)]
        result = np.hstack((result, points))
    return result.astype(int)

def min_dist_to_exemplar(point, cluster_exemplars, data):
    dists = cdist([data[point]], data[cluster_exemplars.astype(np.int32)])
    return dists.min()

def dist_vector(point, exemplar_dict, data):
    result = {}
    for cluster in exemplar_dict:
        result[cluster] = min_dist_to_exemplar(point, exemplar_dict[cluster], data)
    np.array(list(result.values()))
    return np.array(list(result.values()))

def dist_membership_vector(point, exemplar_dict, data, softmax=False):
    if softmax:
        result = np.exp(1./(dist_vector(point, exemplar_dict, data) + 1e-8))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = 1./(dist_vector(point, exemplar_dict, data) + 1e-8)
        result[~np.isfinite(result)] = np.finfo(np.double).max
    result /= result.sum()
    return result

"""
Outlier based membership
"""
def max_lambda_val(cluster, tree):
    cluster_tree = tree[tree['child_size'] > 1]
    leaves = hdbscan.plots._recurse_leaf_dfs(cluster_tree, cluster)
    max_lambda = 0.0
    for leaf in leaves:
        max_lambda = max(max_lambda,
                         tree['lambda_val'][tree['parent'] == leaf].max())
    return max_lambda

def points_in_cluster(cluster, tree):
    leaves = hdbscan.plots._recurse_leaf_dfs(tree, cluster)
    return leaves

def merge_height(point, cluster, tree, point_dict):
    cluster_row = tree[tree['child'] == cluster]
    if point in point_dict[cluster]:
        merge_row = tree[tree['child'] == float(point)][0]
        return merge_row['lambda_val']
    else:
        while point not in point_dict[cluster]:
            parent_row = tree[tree['child'] == cluster]
            cluster = parent_row['parent'].astype(np.float64)[0]
        for row in tree[tree['parent'] == cluster]:
            child_cluster = float(row['child'])
            if child_cluster == point:
                return row['lambda_val']
            if child_cluster in point_dict and point in point_dict[child_cluster]:
                return row['lambda_val']

def per_cluster_scores(point, cluster_ids, tree, max_lambda_dict, point_dict):
    result = {}
    point_row = tree[tree['child'] == point]
    point_cluster = float(point_row[0]['parent'])
    max_lambda = max_lambda_dict[point_cluster] + 1e-8 # avoid zero lambda vals
    
    for c in cluster_ids:
        height = merge_height(point, c, tree, point_dict)
        if max_lambda == np.inf:
            if len(cluster_ids) == 1:
                result[c] = 1.0
            elif height == np.inf:
                result[c] = 0.0
            else:
                result[c] = 1.0
        else:
            result[c] = (max_lambda / (max_lambda - height))
    return result

def outlier_membership_vector(point, cluster_ids, tree,
                              max_lambda_dict, point_dict, softmax=True):
    if softmax:
        result = np.exp(np.array(list(per_cluster_scores(point,
                                                         cluster_ids,
                                                         tree,
                                                         max_lambda_dict,
                                                         point_dict
                                                        ).values())))
        result[~np.isfinite(result)] = np.finfo(np.double).max
    else:
        result = np.array(list(per_cluster_scores(point,
                                                  cluster_ids,
                                                  tree,
                                                  max_lambda_dict,
                                                  point_dict
                                                 ).values()))
    result /= result.sum()
    return result

"""
Combine both
"""
def combined_membership_vector(point, data, tree, exemplar_dict, cluster_ids,
                               max_lambda_dict, point_dict, softmax=False):
    raw_tree = tree._raw_tree
    dist_vec = dist_membership_vector(point, exemplar_dict, data, softmax)
    outl_vec = outlier_membership_vector(point, cluster_ids, raw_tree,
                                         max_lambda_dict, point_dict, softmax)
    result = dist_vec * outl_vec
    result /= result.sum()
    return result

"""
Converting a conditional probability
"""
def prob_in_some_cluster(point, tree, cluster_ids, point_dict, max_lambda_dict):
    heights = []
    for cluster in cluster_ids:
        heights.append(merge_height(point, cluster, tree._raw_tree, point_dict))
    height = max(heights)
    nearest_cluster = cluster_ids[np.argmax(heights)]
    max_lambda = max_lambda_dict[nearest_cluster]

    if max_lambda == np.inf:
        return 1.0
    else:
        return height / max_lambda

"""
Output function
"""
def hdb_fuzzy_clustering(data):
    max_clusters = 5
    min_size = int(data.shape[0]/max_clusters)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, allow_single_cluster=True).fit(data)
    
    # Distance based : membership_vector = dist_membership_vector(x, exemplar_dict, data)
    tree = clusterer.condensed_tree_
    exemplar_dict = {c:exemplars(c,tree) for c in tree._select_clusters()}

    # Outlier based : membership_vector = outlier_membership_vector(x, cluster_ids, raw_tree,
    #                                             max_lambda_dict, point_dict, False)
    cluster_ids = tree._select_clusters()
    raw_tree = tree._raw_tree
    all_possible_clusters = np.arange(data.shape[0], raw_tree['parent'].max() + 1).astype(np.float64)
    max_lambda_dict = {c:max_lambda_val(c, raw_tree) for c in all_possible_clusters}
    point_dict = {c:set(points_in_cluster(c, raw_tree)) for c in all_possible_clusters}

    fuzzy_membership = np.empty((data.shape[0],clusterer.labels_.max()+1))

    for x in range(data.shape[0]):
        membership_vector = combined_membership_vector(x, data, tree, exemplar_dict, cluster_ids, 
                                                       max_lambda_dict, point_dict, False)
        membership_vector *= prob_in_some_cluster(x, tree, cluster_ids, point_dict, max_lambda_dict)
        
        fuzzy_membership[x] = membership_vector
        
    return fuzzy_membership