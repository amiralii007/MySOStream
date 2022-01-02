import operator
import time
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from cluster import Cluster
from micro_cluster import MicroCluster
from SOStream import SOStream


def calculateParity(clusters, dataCal, datasetUniqueLabel):
    totalValue = 0
    for cluster in clusters:
        parts = cluster.data_points
        labels = np.array([])
        for part in parts:
            labels = np.append(labels, dataCal.iloc[part:part + 1, 2])
        unq_label, counts = np.unique(labels, return_counts=True)
        distance = dict(zip(datasetUniqueLabel, counts))
        dominant_count = max(distance.items(), key=operator.itemgetter(1))[1]
        totalValue += float(dominant_count) / cluster.number_data_points
    return (totalValue / len(clusters)) * 100


def run(path):
    purity_list, cluster_list, start_time = dict(), dict(), time.time()

    data = pd.read_csv(path, skiprows=1, header=None).reindex(
        np.random.permutation(pd.read_csv(path, skiprows=1, header=None).index))
    df_label = np.array(data.iloc[:, 2])
    datasetUniqueLabel = np.unique(df_label)
    data_frame = data.drop([2], axis=1)

    sos = SOStream(data=data_frame, cluster_object=Cluster(), alpha=0.1, minPts=3, merge_threshold=2,
                   decay_rate=0.1, fade_threshold=2)

    for i in range(len(data_frame)):
        matrixDataInput = data_frame[i:i + 1]
        t = time.time()
        if not len(sos.get_cluster_obj().get_clusters()) - 1 >= sos.minPts:
            newMicroClusterItems = MicroCluster(number_data_points=1, centroid=matrixDataInput, radius=0,
                                                current_timestamp=t)
            newMicroClusterItems.insert(matrixDataInput.index[0])
            sos.get_cluster_obj().insert(newMicroClusterItems)

        else:
            win = sos.min_distance(matrixDataInput)
            win_neighbors = sos.find_neighbors(win)

            distance = sklearn.metrics.pairwise.euclidean_distances(matrixDataInput, win.centroid)
            if distance <= win.radius:
                win_neighbors = sos.update_cluster(win, matrixDataInput, win_neighbors)

            else:
                newMicroClusterItems = MicroCluster(number_data_points=1, centroid=matrixDataInput, radius=0,
                                                    current_timestamp=t)
                newMicroClusterItems.insert(matrixDataInput.index[0])
                sos.get_cluster_obj().insert(newMicroClusterItems)

            overlap = sos.find_overlap(win, win_neighbors)
            if len(overlap) > 0:
                sos.merge_clusters(win, overlap)

        if i % 50 == 0:
            cluster_list[i] = len(sos.get_cluster_obj().get_clusters())
            if i != 0:
                purity = calculateParity(sos.get_clusters(), data, datasetUniqueLabel)
                purity_list[i] = purity
        if i % 100 == 0:
            sos.fading_all()

    cluster_list[len(data_frame)], purity_list[len(data_frame)] = len(
        sos.get_cluster_obj().get_clusters()), calculateParity(sos.get_clusters(), data, datasetUniqueLabel)

    print("Number of combined clusters: {" + str(sos.get_number_of_merged_clusters()) + "}\n")
    print("Number of dropped clusters: {" + str(sos.get_number_of_faded_clusters()) + "}\n")
    print("maximum number of clusters: {" + str(len(sos.get_cluster_obj().get_clusters())) + "} cluster\n")
    print("Mean Purity: {" + str(np.mean(list(purity_list.values()))) + "}\n")

    print(cluster_list.items())
    y_pos = list(cluster_list.keys())
    plt.ylabel('number of clusters')
    plt.xlabel('number of received data')
    plt.bar(y_pos, cluster_list.values(), width=15, color='g', align='center', alpha=0.25)
    plt.show()

    print(purity_list.items())
    y_pos = list(purity_list.keys())
    plt.ylabel('Purity')
    plt.xlabel('Total number of data row')
    plt.bar(y_pos, purity_list.values(), width=15, color='c', align='center', alpha=0.25)
    plt.show()

    print("\nprogram time: {" + str(time.time() - start_time) + "} sec")


if __name__ == '__main__':
    run('Dataset_2 .csv')
