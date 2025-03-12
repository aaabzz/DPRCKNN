import numpy as np



def MinMaxScaler(data):

    # data: n*d matrix,type -> numpy.ndarray or list

    N = len(data)
    min_component = np.min(data, axis=0)  # 1*d
    max_component = np.max(data, axis=0)  # 1*d
    max_min = max_component - min_component
    min_component = np.tile(min_component, (N, 1))
    max_min = np.tile(max_min, (N, 1))
    data = (data - min_component) / max_min

    return data

def getNumc(labels):

    labels = np.array(labels)

    return len(np.unique(labels))

def Euclidean_dist(X, Y):

    dist = np.linalg.norm(X - Y)

    return dist

def get_dist_matrix(data):

    num, dim = data.shape
    distmat = np.zeros([num, num])

    for x in range(num):
        for y in range(x + 1, num):

            dist = Euclidean_dist(data[x, :], data[y, :])
            distmat[x, y] = dist
            distmat[y, x] = dist

    return distmat

def KnnSearch(data, distmat, k):

    num = len(data)
    dataknnset = []
    distknnset = []

    for n in range(num):

        dists = distmat[n, :]
        dataknn = np.argsort(dists)[:k+1]
        distknn = np.sort(dists)[:k+1]

        dataknn = dataknn[dataknn != n]
        distknn = distknn[distknn != 0]
        dataknnset.append(dataknn)
        distknnset.append(distknn)

    return dataknnset, distknnset



