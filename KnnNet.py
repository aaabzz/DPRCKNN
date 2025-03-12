import os
import matplotlib.pyplot as plt
from general import *
import numpy as np


# visited

def KnnSearchClusters(clusters, distmat, visited, n_neighbor=7):

    # clusters -> list

    num = len(distmat)
    cluster_dataknnset = []
    cluster_distknnset = []
    done = np.where(visited == 1)[0]

    for cluster in clusters:

        dist_candidates = []
        cluster_done = np.union1d(done, cluster)
        remaining = np.arange(num)
        remaining = np.delete(remaining, cluster_done)
        cluster_dataknn = []
        cluster_distknn = []

        for data in cluster:

            dists = distmat[data, :].copy()
            dists = np.delete(dists, cluster_done)
            dist_candidates.append(dists)

        dist_candidates = np.array(dist_candidates)

        for n in range(n_neighbor):

            dist = np.min(dist_candidates)
            index = np.argmin(dist_candidates)
            index = np.unravel_index(index, dist_candidates.shape)[-1]
            data_index = remaining[index]

            if len(dist_candidates.shape) == 1:

                dist_candidates[index] = np.inf

            else:

                dist_candidates[:, index] = np.ones(dist_candidates.shape[0]) * np.inf

            cluster_distknn.append(dist)
            cluster_dataknn.append(data_index)

        cluster_distknnset.append(cluster_distknn)
        cluster_dataknnset.append(cluster_dataknn)

    return np.array(cluster_dataknnset), np.array(cluster_distknnset)


def KnnSearchCluster(cluster, dataknnset, high_points):

    neighbors = []

    # cluster

    for i in cluster:

        dataknn = dataknnset[i].copy()
        dataknn = dataknn[~np.isin(dataknn, cluster)]
        dataknn = dataknn[np.isin(dataknn, high_points)]

        for j in dataknn:

            neighbors.append(j)

    # high_points

    other_points = high_points[~np.isin(high_points, cluster)]

    for i in other_points:

        dataknn = dataknnset[i].copy()

        if np.isin(dataknn, cluster).any():

            neighbors.append(i)

    return np.unique(np.array(neighbors))

def cal_neighbor_dists(cluster, neighbors, distmat):

    dists = []

    for i in neighbors:

        dist = distmat[cluster[0], i]

        for j in cluster:

            temp = distmat[j, i]

            if temp < dist:

                dist = temp

        dists.append(dist)

    return np.array(dists)

def KnnSearchPoint(distmat, n_neighbor=7):

    num = len(distmat)
    dataknnset = []
    distknnset = []

    for n in range(num):

        dists = distmat[n, :].copy()
        dists[n] = np.inf
        dataknn = np.argsort(dists)[:n_neighbor]
        distknn = np.sort(dists)[:n_neighbor]

        # dataknn = dataknn[dataknn != n]
        # distknn = distknn[distknn != 0]
        dataknnset.append(dataknn)
        distknnset.append(distknn)

    return np.array(dataknnset), np.array(distknnset)

def cal_Density_Matrix(distknnset):

    # maxdists = np.max(distknnset, axis=1)
    # n = len(maxdists)
    # rhoset = []
    #
    # for i in range(n):
    #
    #     rho = 0
    #     dists = distmat[i]
    #     maxdist = maxdists[i]
    #     num = np.where(dists <= maxdist)[0]
    #     for j in num:
    #
    #         rho = rho + np.exp(- (dists[j] ** 2 ) / 2)
    #
    #     rhoset.append(rho)

    n = len(distknnset)
    rhoset = []

    for i in range(n):

        rho = 0
        dists = distknnset[i, :]

        for dist in dists:

            rho = rho + np.exp(-1 * (dist ** 2) / 2)

        rhoset.append(rho)

    return np.array(rhoset)

def get_Mixed_Matrix(cluster_rhoset, rhoset, cluster_dataknnset, cluster_distknnset):

    num = len(cluster_rhoset)
    cluster_mixedset = []

    for n in range(num):

        cluster_dataknn = cluster_dataknnset[n]
        cluster_distknn = cluster_distknnset[n].copy()
        cluster_rho = cluster_rhoset[n]
        rhoknn = rhoset[cluster_dataknn].copy()
        rhoknn = rhoknn / cluster_rho
        rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
        mixedknn = 1 / cluster_rho * rhoknn * cluster_distknn
        # mixedknn = 1 / cluster_rho * (cluster_distknn + rhoknn)
        cluster_mixedset.append(mixedknn)

    return np.array(cluster_mixedset)

# Density as a fraction of distance

def get_Dissimilarity_Matrix(cluster_rhoset, rhoset, cluster_dataknnset, cluster_distknnset):

    num = len(cluster_rhoset)
    cluster_mixedset = []

    for n in range(num):

        cluster_dataknn = cluster_dataknnset[n]
        cluster_distknn = cluster_distknnset[n].copy()
        cluster_rho = cluster_rhoset[n]
        rhoknn = rhoset[cluster_dataknn].copy()
        rhoknn = rhoknn / cluster_rho
        rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
        mixedknn = rhoknn * cluster_distknn
        # mixedknn = 1 / cluster_rho * (cluster_distknn + rhoknn)
        cluster_mixedset.append(mixedknn)

    return np.array(cluster_mixedset)

def cal_Cluster_Dissimilarity(rho, rhoset, dataknn, distknn):

    rhoknn = rhoset[dataknn].copy()
    rhoknn = rhoknn / rho
    rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
    mixedknn = 1 / rho * rhoknn * distknn
    # mixedknn = rhoknn * distknn
    # mixedknn = 1 / rho * (distknn + rhoknn)

    return np.array(mixedknn)

def get_Current_sort(mixedset):

    sorted_flat_indices = np.argsort(mixedset, axis=None)

    sorted_2d_indices = np.unravel_index(sorted_flat_indices, mixedset.shape)

    sorted_2d_indices = np.array(sorted_2d_indices).T

    return sorted_2d_indices  # ndarray

def get_Current_index(mixedset):

    sorted_flat_index = np.argmin(mixedset)

    sorted_2d_index = np.unravel_index(sorted_flat_index, mixedset.shape)

    return sorted_2d_index

# inflection

def cal_inflection():

    pass


