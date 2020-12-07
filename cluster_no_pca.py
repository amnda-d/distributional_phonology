import numpy as np
from math import exp, log, pi
from sklearn.cluster import KMeans
from scipy.spatial import distance
from pca import get_pcs

def find_classes(ppmi, vocab, classes = set(),  max_k = 3, max_pcs = None):
    pcas=ppmi
    if pcas.shape[0] == 0 or pcas.shape[0] == 1:
        return classes
    vocab_idx = {v:k for k, v in vocab.items()}
    for i in range(pcas.shape[1]):
        col = pcas[:, i].reshape(-1, 1)
        best_kmeans = cluster(col, max_k)
        clusters = [
            tuple(sorted([vocab_idx[x] for x in np.where(best_kmeans.labels_ == y)[0]])) for y in range(best_kmeans.n_clusters)]
        for c in clusters:
            if c not in classes:
                classes.add(c)
                # print('\tnew class: ', c)
                if len(c) > 1:
                    c_idx = [vocab[x] for x in c]
                    subset = ppmi[c_idx, :]
                    subvocab = { k: v for v, k in enumerate(sorted(c))}
                    classes.update(find_classes(subset, subvocab, set([tuple(subvocab.keys())])))
    return classes

def cluster(pca_col, max_k):
    kmeans = None
    max_bic = -np.inf
    best_kmeans = None
    for k in range(1, min(pca_col.shape[0], max_k) + 1):
        kmeans = KMeans(n_clusters = k)
        clustering = kmeans.fit(pca_col)
        cl_bic = compute_bic(clustering, pca_col)
        if cl_bic > max_bic or best_kmeans is None:
            max_bic = cl_bic
            best_kmeans = clustering
    return best_kmeans

def calculate_mean_and_variance(X, n):
    '''
    Calculate the mean and variance of a cluster
    '''
    my_sum = 0
    sumsq = 0

    sorted_X = sorted(X)

    median = sorted_X[len(sorted_X) // 2]
    for item in sorted_X:
        my_sum += item - median
        sumsq += (item - median) * (item - median)
    mean = my_sum / n + median

    if n > 1:
        variance = (sumsq - my_sum * my_sum / n) / (n - 1)
    else:
        variance = 0

    return mean, variance

def compute_bic(kmeans, X):
    """
    From https://github.com/connormayer/distributional_learning/blob/master/code/clusterer.py#L66

    Implementing BIC as defined results in an error when number of elements = number of clusters. Mayer's solution is to replace variance with the minimum distance to a point in another cluster.
    """
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)

    if len(n) != m:
        # We've got a cluster with 0 elements in it. This rarely happens, but
        # should result in infinite BIC.
        return -np.inf

    lamb = []
    coeff = []
    means = []
    sigmas = []

    for k in range(m):
        lamb.append(n[k] / len(labels))
        mean, variance = calculate_mean_and_variance(X[np.where(labels == k)], n[k])
        means.append(mean)
        sigmas.append(variance)

        # If we can't calculate variance within cluster, use minimum distance to point
        # in other cluster.
        if variance == 0 or n[k] == 1:
            X_sorted = np.copy(X).reshape(1, -1)[0]
            X_sorted.sort()
            left = min(np.where(X_sorted == means[k])[0]) - 1
            right = max(np.where(X_sorted == means[k])[0]) + 1

            if right >= len(X):
                dmin = X_sorted[left + 1] - X_sorted[left]
            elif left < 0:
                dmin = X_sorted[right] - X_sorted[right - 1]
            else:
                dmin = min(
                    X_sorted[left + 1] - X_sorted[left],
                    X_sorted[right] - X_sorted[right - 1]
                )

            if variance == 0:
                sigmas[-1] = dmin * dmin / 4.0 / 9.0
            if n[k] == 1:
                sigmas[-1] = dmin * dmin

        coeff.append(lamb[k] / (2.0 * pi * sigmas[-1])**0.5)

    log_likelihood = 0
    for item in X:
        likelihood = 0
        for k in range(m):
            likelihood += coeff[k] * exp(-(item - means[k]) * (item - means[k]) / (2 * sigmas[k]))
        log_likelihood += log(likelihood)

    # (3 * m - 1) is used because each of the clusters has three associated parameters:
    # * the cluster centroid coordinate
    # * the cluster variance
    # * the cluster probability
    # The -1 is because the probabilities must sum to 1, so there are only m-1 free probs.
    bic = 2 * log_likelihood - (3 * m - 1) * log(len(X))
    return bic

def bic(kmeans, X):
    # from https://stats.stackexchange.com/a/251169
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / ((N - m) / d)) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)]) if (N-m) > 0 else 0.0


    const_term = 0.5 * m * np.log(N) * (d+1)

    return np.sum([
        n[i] * np.log(n[i]) -
        n[i] * np.log(N) -
        ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
        ((n[i] - 1) * d/ 2)
        for i in range(m)
    ]) - const_term
