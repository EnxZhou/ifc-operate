import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.cluster import SpectralClustering


def main():
    G = nx.read_graphml("dataset/NXJ4-NXZ_8_1.graphml")

    X = nx.to_numpy_array(G)
    sim_matrix = rbf_kernel(X, gamma=0.1)

    n_clusters = 10
    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity='precomputed',
                            assign_labels='discretize')
    labels = sc.fit_predict(sim_matrix)

    for i, node in enumerate(G.nodes):
        G.nodes[node]['predict_label'] = labels[i]

    nx.write_graphml(G, "dataset/NXJ4-NXZ_8_1_labeled.graphml")


if __name__ == '__main__':
    main()
