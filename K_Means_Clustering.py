import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf

def cluster(data_dir):
    # Get data from saved csv
    trainX_df = pd.read_csv(os.path.join(data_dir, "train_X"))
    train_X = np.array(trainX_df)

    num_clusters = 5
    kmeans = tf.compat.v1.estimator.experimental.KMeans(num_clusters=num_clusters)

    def input_fn():
        return tf.compat.v1.train.limit_epochs(
            tf.convert_to_tensor(train_X, dtype=tf.float32), num_epochs=1)

    prev_centers = None
    for _ in range(20):
        kmeans.train(input_fn)
        cluster_centers = kmeans.cluster_centers()
        if prev_centers is not None:
            print('delta:', cluster_centers - prev_centers)
        prev_centers = cluster_centers
        print('score:', kmeans.score(input_fn))
    print('cluster centers:', cluster_centers)

    cluster_indices = list(kmeans.predict_cluster_index(input_fn))
    label_dict = {}
    for i, point in enumerate(train_X):
        ci = cluster_indices[i]
        center = cluster_centers[ci]
        label_dict[point] = center

    return label_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required = True)
    args = parser.parse_args()
    cluster(args.data_dir)