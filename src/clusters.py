from sklearn.cluster import *


def paperCluster_offical(dis, name_papers, n_cluster, method='AG',linkage='ward',affinity="euclidean"):
    if method == 'AG':

        if linkage == 'ward':

            cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage= linkage, affinity='euclidean')

        else:
            cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage= linkage, affinity=affinity)

    elif method == 'AP':
        cmodel = AffinityPropagation(damping=0.5, affinity='precomputed')
    else:
        cmodel = DBSCAN(eps=0.25, min_samples=5, metric='precomputed')



    indexs = cmodel.fit_predict(dis)
    result = []

    separates = []
    for i, value in enumerate(indexs):
        if i >= len(name_papers):
            break

        if value == -1:
            separates.append(name_papers[i])
            continue

        while value >= len(result):
             result.append([])

        result[value].append(name_papers[i])

    if len(separates) > 0:
        result.append(separates)

    return result


def paperClusterByDis(dis, name_papers, n_cluster, method='AG',linkage='average'):
    if method == 'AG':
        cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage= linkage, affinity='precomputed')
    elif method == 'AP':
        cmodel = AffinityPropagation(damping=0.5, affinity='precomputed')
    else:
        cmodel = DBSCAN(eps=0.25, min_samples=5, metric='precomputed')
    indexs = cmodel.fit_predict(dis)
    result = []

    separates = []
    for i, value in enumerate(indexs):
        if i >= len(name_papers):
            break

        if value == -1:
            separates.append(name_papers[i])
            continue

        while value >= len(result):
             result.append([])

        result[value].append(name_papers[i])

    if len(separates) > 0:
        result.append(separates)

    return result


def paperCluster(input, name_papers, n_cluster, method='euclidean'):
    if method == 'euclidean':
        cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage='average')
    else:
        cmodel = AgglomerativeClustering(n_clusters=n_cluster, linkage='average', affinity='cosine')

        # cmodel = AgglomerativeClustering(n_clusters=None, distance_threshold=0.8, linkage='average', affinity='cosine')

    indexs = cmodel.fit_predict(input)
    result = []

    for i, value in enumerate(indexs):
        if i >= len(name_papers):
            break

        while value >= len(result):
             result.append([])

        result[value].append(name_papers[i])

    return result


def paperSpectraCluster(X, name_papers, n_cluster):
    SC = SpectralClustering(n_cluster, affinity='precomputed', n_init=100, assign_labels='discretize')
    indexs = SC.fit_predict(X)
    result = [[] for _ in range(0, n_cluster)]

    for i, value in enumerate(indexs):
        if i >= len(name_papers):
            break
        result[value].append(name_papers[i])

    return result


if __name__ == '__main__':
    pass
