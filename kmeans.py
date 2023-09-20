import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def plot_elbow(data, max_clusters=10, random_state=None, n_init=10):
    inertia_values = []
    # Performing the KMeans algorithm for various numbers of clusters.
    for k in range(1, max_clusters + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        km.fit(data)
        inertia_values.append(km.inertia_)

    # Plotting the Inertia value as a graph.
    plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()