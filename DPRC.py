import matplotlib.pyplot as plt
import numpy as np
import math
from KnnNet import *
from general import *
import threading
from colormap import color_mapping


def get_clicked_x(x, y):

    click_x = None
    cilck_event = threading.Event()

    def on_click(event):

        nonlocal click_x

        if event.inaxes:

            click_x = event.xdata
            click_x = math.floor(click_x)
            cilck_event.set()
            plt.close()

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', linestyle='--')
    ax.set_title("Click to get X-axis value and close the plot")
    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()
    cilck_event.wait()

    return click_x

class DensityPeakRegionCluster(object):

    def __init__(self,
                 n_clusters=None,
                 n_neighbor=7,
                 max_DPR = False,
                 DPR_visualizaiton = False,
                 cluster_visualization = False,
                 density_group=False):

        self.n_clusters = n_clusters
        self.n_neighbor = n_neighbor

        self.max_DPR = max_DPR
        self.DPR_visualization = DPR_visualizaiton
        self.cluster_visualizaiton = cluster_visualization
        self.density_group = density_group

    def build_distance(self):

        datanorm = MinMaxScaler(self.data)
        distmat = get_dist_matrix(datanorm)

        return distmat

    def KnnsearchPoint(self):

        dataknnset = []
        distknnset = []

        for n in range(self.num):
            dists = self.distmat[n, :].copy()
            dists[n] = np.inf
            dataknn = np.argsort(dists)[:self.n_neighbor]
            distknn = np.sort(dists)[:self.n_neighbor]

            # dataknn = dataknn[dataknn != n]
            # distknn = distknn[distknn != 0]
            dataknnset.append(dataknn)
            distknnset.append(distknn)

        return np.array(dataknnset), np.array(distknnset)

    def KnnSearchCluster(self, cluster, high_points):

        neighbors = []

        # cluster

        for i in cluster:

            dataknn = self.dataknnset[i].copy()
            dataknn = dataknn[~np.isin(dataknn, cluster)]
            dataknn = dataknn[np.isin(dataknn, high_points)]

            for j in dataknn:
                neighbors.append(j)

        # high_points

        other_points = high_points[~np.isin(high_points, cluster)]

        for i in other_points:

            dataknn = self.dataknnset[i].copy()

            if np.isin(dataknn, cluster).any():
                neighbors.append(i)

        return np.unique(np.array(neighbors))

    def cal_neighbor_dists(self, cluster, neighbors):

        dists = []

        for i in neighbors:

            dist = self.distmat[cluster[0], i]

            for j in cluster:

                temp = self.distmat[j, i]

                if temp < dist:
                    dist = temp

            dists.append(dist)

        return np.array(dists)

    def KnnSearchClusters(self, clusters, visited):

        # clusters -> list

        cluster_dataknnset = []
        cluster_distknnset = []
        done = np.where(visited == 1)[0]

        for cluster in clusters:

            dist_candidates = []
            cluster_done = np.union1d(done, cluster)
            remaining = np.arange(self.num)
            remaining = np.delete(remaining, cluster_done)
            cluster_dataknn = []
            cluster_distknn = []

            for data in cluster:
                dists = self.distmat[data, :].copy()
                dists = np.delete(dists, cluster_done)
                dist_candidates.append(dists)

            dist_candidates = np.array(dist_candidates)

            for n in range(self.n_neighbor):

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

    def cal_Density_Matrix(self):

        n = len(self.distknnset)
        rhoset = []

        for i in range(n):

            rho = 0
            dists = self.distknnset[i, :]

            for dist in dists:
                rho = rho + np.exp(-1 * (dist ** 2) / 2)

            rhoset.append(rho)

        rhoset = np.array(rhoset)
        rho_aver = np.sum(rhoset) / self.num
        high_points = np.where(rhoset >= rho_aver)[0]
        low_points = np.where(rhoset < rho_aver)[0]

        return np.array(rhoset), high_points, low_points

    def cal_Cluster_Dissimilarity(self, rho, dataknn, distknn):

        rhoknn = self.rhoset[dataknn].copy()
        rhoknn = rhoknn / rho
        rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
        mixedknn = 1 / rho * rhoknn * distknn
        # mixedknn = rhoknn * distknn
        # mixedknn = 1 / rho * (distknn + rhoknn)

        return np.array(mixedknn)

    def get_Mixed_Matrix(self, cluster_rhoset, cluster_dataknnset, cluster_distknnset):

        num = len(cluster_rhoset)
        cluster_mixedset = []

        for n in range(num):
            cluster_dataknn = cluster_dataknnset[n]
            cluster_distknn = cluster_distknnset[n].copy()
            cluster_rho = cluster_rhoset[n]
            rhoknn = self.rhoset[cluster_dataknn].copy()
            rhoknn = rhoknn / cluster_rho
            rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
            mixedknn = 1 / cluster_rho * rhoknn * cluster_distknn
            # mixedknn = 1 / cluster_rho * (cluster_distknn + rhoknn)
            cluster_mixedset.append(mixedknn)

        return np.array(cluster_mixedset)

    def get_Dissimilarity_Matrix(self, cluster_rhoset, cluster_dataknnset, cluster_distknnset):

        num = len(cluster_rhoset)
        cluster_mixedset = []

        for n in range(num):
            cluster_dataknn = cluster_dataknnset[n]
            cluster_distknn = cluster_distknnset[n].copy()
            cluster_rho = cluster_rhoset[n]
            rhoknn = self.rhoset[cluster_dataknn].copy()
            rhoknn = rhoknn / cluster_rho
            rhoknn = -1 * np.exp(-1 * ((rhoknn - 1) ** 2) / 2) + 2
            mixedknn = rhoknn * cluster_distknn
            # mixedknn = 1 / cluster_rho * (cluster_distknn + rhoknn)
            cluster_mixedset.append(mixedknn)

        return np.array(cluster_mixedset)

    def get_Current_sort(self, mixedset):

        sorted_flat_indices = np.argsort(mixedset, axis=None)

        sorted_2d_indices = np.unravel_index(sorted_flat_indices, mixedset.shape)

        sorted_2d_indices = np.array(sorted_2d_indices).T

        return sorted_2d_indices  # ndarray

    def get_Current_index(self, mixedset):

        sorted_flat_index = np.argmin(mixedset)

        sorted_2d_index = np.unravel_index(sorted_flat_index, mixedset.shape)

        return sorted_2d_index

    def extract_DPRs_auto(self):

        high_points = self.high_points.copy()
        low_points = self.low_points.copy()
        DPRs = []
        visited = np.zeros(self.num, dtype=int)

        if self.n_clusters is None:

            raise ValueError('Please give the parameter n_clusters.')

        if self.n_clusters <= 0:

            raise ValueError('The parameter n_clusters must be greater than 0.')

        for i in range(self.n_clusters):

            if np.any(visited[high_points] != 1):

                high_rhoset = self.rhoset[high_points]
                high_dataknnset = self.dataknnset[high_points]
                high_distknnset = self.distknnset[high_points]

                mixed = self.get_Mixed_Matrix(high_rhoset, high_dataknnset, high_distknnset)
                sorted_2d_index = self.get_Current_index(mixed)
                row, col = sorted_2d_index
                point = high_points[row]
                another_point = high_dataknnset[row, col]
                DPR = []

                if visited[point] != 1:

                    DPR.append(point)

                if visited[another_point] != 1 and another_point in high_points:

                    DPR.append(another_point)

                while True:

                    neighbors = self.KnnSearchCluster(DPR, high_points)

                    if neighbors.size == 0:

                        break

                    if np.all(visited[neighbors] == 1):

                        break

                    dists = self.cal_neighbor_dists(DPR, neighbors)
                    rho = np.sum(self.rhoset[DPR]) / len(DPR)
                    mixedknn = self.cal_Cluster_Dissimilarity(rho, neighbors, dists)
                    indices = np.argsort(mixedknn)

                    for index in indices:

                        point = neighbors[index]

                        if visited[point] != 1:

                            DPR.append(point)
                            break

                        else:

                            continue

                DPRs.append(DPR)
                visited[DPR] = 1
                high_points = high_points[~np.isin(high_points, DPR)]

            else:

                low_rhoset = self.rhoset[low_points]
                low_dataknnset = self.dataknnset[low_points]
                low_distknnset = self.distknnset[low_points]
                mixed = self.get_Mixed_Matrix(low_rhoset, low_dataknnset, low_distknnset)
                sorted_2d_index = get_Current_index(mixed)
                row, col = sorted_2d_index
                point = low_points[row]
                another_point = low_dataknnset[row, col]
                DPR = [point, another_point]
                DPRs.append(DPR)
                visited[DPR] = 1
                low_points = low_points[~np.isin(low_points, DPR)]

        return DPRs, visited

    def extract_DPRs_manual(self):

        high_points = self.high_points.copy()
        low_points = self.low_points.copy()
        DPRs = []
        visited = np.zeros(self.num, dtype=int)

        if self.n_clusters is None:
            raise ValueError('Please give the parameter n_clusters.')

        if self.n_clusters <= 0:
            raise ValueError('The parameter n_clusters must be greater than 0.')

        for i in range(self.n_clusters):

            if np.any(visited[high_points] != 1):

                high_rhoset = self.rhoset[high_points]
                high_dataknnset = self.dataknnset[high_points]
                high_distknnset = self.distknnset[high_points]

                mixed = self.get_Mixed_Matrix(high_rhoset, high_dataknnset, high_distknnset)
                sorted_2d_index = self.get_Current_index(mixed)
                row, col = sorted_2d_index
                point = high_points[row]
                another_point = high_dataknnset[row, col]
                dissimilarity = mixed[row, col]
                DPR = []
                dissimilarities = []

                if visited[point] != 1:
                    DPR.append(point)

                if visited[another_point] != 1 and another_point in high_points:
                    DPR.append(another_point)
                    dissimilarities.append(dissimilarity)

                while True:

                    neighbors = self.KnnSearchCluster(DPR, high_points)

                    if neighbors.size == 0:
                        break

                    if np.all(visited[neighbors] == 1):
                        break

                    dists = self.cal_neighbor_dists(DPR, neighbors)
                    rho = np.sum(self.rhoset[DPR]) / len(DPR)
                    mixedknn = self.cal_Cluster_Dissimilarity(rho, neighbors, dists)
                    indices = np.argsort(mixedknn)

                    for index in indices:

                        point = neighbors[index]
                        dissimilarity = mixedknn[index]

                        if visited[point] != 1:

                            DPR.append(point)
                            dissimilarities.append(dissimilarity)
                            break

                        else:

                            continue

                x = np.arange(len(dissimilarities))
                x_value = get_clicked_x(x, dissimilarities)
                x_value = x_value + 1

                if x_value < len(DPR):

                    DPR = DPR[:x_value]

                DPRs.append(DPR)
                visited[DPR] = 1
                high_points = high_points[~np.isin(high_points, DPR)]

            else:

                low_rhoset = self.rhoset[low_points]
                low_dataknnset = self.dataknnset[low_points]
                low_distknnset = self.distknnset[low_points]
                mixed = self.get_Mixed_Matrix(low_rhoset, low_dataknnset, low_distknnset)
                sorted_2d_index = get_Current_index(mixed)
                row, col = sorted_2d_index
                point = low_points[row]
                another_point = low_dataknnset[row, col]
                DPR = [point, another_point]
                DPRs.append(DPR)
                visited[DPR] = 1
                low_points = low_points[~np.isin(low_points, DPR)]

        return DPRs, visited

    def neighbor_assignment(self, visited):

        clusters = self.DPRs.copy()

        for i in range(len(visited[visited != 1])):

            cluster_rhoset = []

            for i in clusters:
                rho = np.sum(self.rhoset[i]) / len(i)
                cluster_rhoset.append(rho)

            cluster_rhoset = np.array(cluster_rhoset)

            cluster_dataknnset, cluster_distknnset = self.KnnSearchClusters(clusters, visited)
            cluster_mixed = self.get_Dissimilarity_Matrix(cluster_rhoset, cluster_dataknnset, cluster_distknnset)
            sorted_2d_indices = get_Current_sort(cluster_mixed)

            for row, col in sorted_2d_indices:

                c = clusters[row]
                point = cluster_dataknnset[row, col]

                if visited[point] == 1:

                    continue

                else:

                    visited[point] = 1
                    c.append(point)
                    break

        return clusters

    def get_labels(self):

        labels = np.zeros(self.num, dtype=int)

        for i in range(len(self.clusters)):

            labels[self.clusters[i]] = i + 1

        return labels

    def plot_density_group(self):

        plt.scatter(self.data[self.high_points, 0], self.data[self.high_points, 1], c='blue', s=20)
        plt.scatter(self.data[self.low_points, 0], self.data[self.low_points, 1], c='red', s=20)
        plt.show()

    def plot_DPR(self):

        labels = np.zeros(self.num, dtype=int)

        for i in range(len(self.DPRs)):

            DPR = self.DPRs[i]
            labels[DPR] = i + 1

        label_color = []

        for i in range(len(labels)):

            if labels[i] == 0:

                label_color.append('black')

            else:

                label_color.append(color_mapping[labels[i]])

        plt.scatter(self.data[:, 0], self.data[:, 1], c=label_color, s=20)
        plt.show()

    def plot_clusters(self):

        label_color = [color_mapping[i] for i in self.labels]
        plt.scatter(self.data[:, 0], self.data[:, 1], c=label_color, s=20)

        for DPR in self.DPRs:

            point = DPR[0]
            plt.scatter(self.data[point, 0], self.data[point, 1], c=label_color[point], edgecolors='black', marker='*', s=50)

        plt.show()


    def fit(self, data):

        self.data = data
        self.num = len(data)
        self.distmat = self.build_distance()
        self.dataknnset, self.distknnset = self.KnnsearchPoint()
        self.rhoset, self.high_points, self.low_points = self.cal_Density_Matrix()

        if self.density_group:

            self.plot_density_group()

        if self.max_DPR:

            self.DPRs, visited = self.extract_DPRs_manual()

        else:

            self.DPRs, visited = self.extract_DPRs_auto()

        if self.DPR_visualization:

            self.plot_DPR()

        self.clusters = self.neighbor_assignment(visited)

        self.labels = self.get_labels()

        if self.cluster_visualizaiton:

            self.plot_clusters()

        return self.labels



