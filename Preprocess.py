import numpy as np
import pandas as pd
import tensorflow as tf
import os
import argparse
from utils import labels

def preprocess_data(data_dir, output_dir):
    allfiles = os.listdir(data_dir)
    npzfiles = []
    for idx, f in enumerate(allfiles):
        if ".npz" in f:
            npzfiles.append(os.path.join(data_dir, f))
    npzfiles.sort()
    print("{} total files found and downloaded...".format(len(npzfiles)))

    # Randomly split in 90% train, 10% test
    idx = np.random.permutation(len(npzfiles))
    train_idx = idx[: 9*len(npzfiles) // 10]
    test_idx = idx[9*len(npzfiles) // 10 :]
    print("Randomly generated a training set of {} files...".format(len(train_idx)))
    print("Randomly generated a test set of {} files...".format(len(test_idx)))

    # Extract train and test files
    train_files = [npzfiles[i] for i in train_idx]
    test_files = [npzfiles[i] for i in test_idx]
    train_X, train_y = extract_files(train_files)
    test_X, test_y = extract_files(test_files)
    
    # Save processed data to output directory
    np.save(os.path.join(output_dir, "train_X.npy"), train_X ,allow_pickle=False)
    np.save(os.path.join(output_dir, "train_y.npy"), train_y ,allow_pickle=False)
    np.save(os.path.join(output_dir, "test_X.npy"), test_X ,allow_pickle=False)
    np.save(os.path.join(output_dir, "test_y.npy"), test_y ,allow_pickle=False)

def extract_files(files):
    data = []
    labels = []
    fs = None
    for file in files:
        print("Loading {} ...".format(file))
        with np.load(file) as f:
            d = f['x']
            l = f['y']
            sr = f['fs']
        fs = sr
        data.append(d)
        labels.append(l)
    data = np.vstack(data)
    labels = np.hstack(labels)
    labels = tf.one_hot(labels, 5)
    return data, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, required = True)
    parser.add_argument("--output_dir", type = str, required = True)
    args = parser.parse_args()

    preprocess_data(args.data_dir, args.output_dir)