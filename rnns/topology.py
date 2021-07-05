import numpy as np

from bct.algorithms import(centrality, clustering, degree, distance, modularity, core, similarity)
from scipy.spatial.distance import cdist

import networkx as nx
from networkx.algorithms import clique


def local_topology(w, property):
    _, _, n = w.shape
    return np.vstack([local_properties(property, w[:,:,i]) for i in range(n)])


def local_properties(prop, conn):

    conn_wei = conn.copy()
    conn_bin = conn_wei.copy().astype(bool).astype(int)

    if prop == 'node_strength': #***
        property = degree.strengths_und(conn_wei)

    elif prop == 'node_degree': #***
        property = degree.degrees_und(conn_bin)

    elif prop == 'wei_clustering_coeff': #***
        property = clustering.clustering_coef_wu(conn_wei)

    elif prop == 'bin_clustering_coeff': #***
        property = clustering.clustering_coef_bu(conn_bin)

    elif prop == 'wei_centrality':
        N = len(conn)
        property = centrality.betweenness_wei(1/conn_wei)/((N-1)*(N-2))

    elif prop ==  'bin_centrality':
        N = len(conn)
        property = centrality.betweenness_bin(conn_bin)/((N-1)*(N-2))

    elif  prop == 'wei_participation_coeff': #***
        property = centrality.participation_coef(conn_wei, ci=class_mapping)

    elif  prop == 'bin_participation_coeff': #***
        property = centrality.participation_coef(conn_bin, ci=class_mapping)

    elif  prop == 'wei_diversity_coeff': #***
        pos, _ = centrality.diversity_coef_sign(conn_wei, ci=class_mapping)
        property = pos

    return property[np.newaxis, :]
