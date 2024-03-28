import numpy as np
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# First attempt at completely processing, vectorizing, and classifying data
# This was the jump off point for our more intricate pre and post processing
# files.

labels_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}

# Read in all files from directory
data_dir = "data/eeg_fpz_cz"
allfiles = os.listdir(data_dir)
print(allfiles)

# Process only .npz files
npzfiles = []
for idx, f in enumerate(allfiles):
    if ".npz" in f:
        npzfiles.append(os.path.join(data_dir, f))
npzfiles.sort()
print(npzfiles)

# Randomly split in 90% train, 10% test
idx = np.random.permutation(len(npzfiles))
train_idx = idx[: 9*len(npzfiles) // 10]
test_idx = idx[9*len(npzfiles) // 10 :]

# Extract train and test files
train_files = [npzfiles[i] for i in train_idx]
test_files = [npzfiles[i] for i in test_idx]

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
    return data, labels

train_X, train_y = extract_files(train_files)
test_X, test_y = extract_files(test_files)

# Create 'dummy' model for proof of concept
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 3000, tol = 1e-3, verbose = 2)
model.fit(train_X[:,:,0], train_y[:])

test_yhat = model.predict(test_X[:,:,0])

def calc_accuracy(yhat, ytrue):
    assert len(yhat) == len(ytrue), "Vector sizes do not match."
    count = 0
    for i in range(len(yhat)):
        if yhat[i] == ytrue[i]:
            count += 1
    return count / len(yhat)

print(calc_accuracy(test_yhat, test_y))

data,_ = extract_files([npzfiles[1]])
data_hat = model.predict(data[:,:,0])

plt.figure(figsize=(15,5))
plt.plot(range(1, len(data_hat)+1), data_hat)
plt.title('Hypnogram of Automatically Scored Data')
plt.ylabel('Sleep Stage')
plt.xlabel('Epoch (30s each)')
plt.show()