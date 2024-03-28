import numpy as np
from sklearn.model_selection import learning_curve
import tensorflow as tf
import os
import argparse
from tensorflow import keras
from CNN_Model import CNNModel
from CNN_LSTM_model import CNN_LSTM
from CNN_Transformer_Model import CNN_Transformer

def report_statistics(model_path, data_path, labels_path):
    test_X = np.genfromtxt(data_path, delimiter=',')
    test_y = np.genfromtxt(labels_path, delimiter=',')

    model = tf.keras.models.load_model(model_path)
    y_hat = model.predict(test_X)
    print("Calculated model accuracy is: {}".format(calc_accuracy(y_hat, test_y)))
    print("Calculated model accuracy is: {}".format(calc_F1(y_hat, test_y)))

def calc_F1(yhat, ytrue):
    true_p = np.array([0,0,0,0,0])
    false_p = np.array([0,0,0,0,0])
    false_n = np.array([0,0,0,0,0])
    for i in range(len(yhat)):
        if yhat[i] == ytrue[i]:
            true_p[yhat[i]] += 1
        else:
            false_p[yhat[i]] += 1
            false_n[ytrue[i]] += 1
    precisions = [true_p[i]/(true_p[i]+false_p[i]) for i in range(5)]
    recalls = [true_p[i]/(true_p[i]+false_n[i]) for i in range(5)]
    F1s = [2*(precisions[i]*recalls[i])/(precisions[i]+recalls[i]) for i in range(5)]
    return np.mean(F1s)


def calc_accuracy(yhat, ytrue):
    assert len(yhat) == len(ytrue), "Vector sizes do not match."
    count = 0
    for i in range(len(yhat)):
        if np.argmax(yhat[i])==np.argmax(ytrue[i]):
            count += 1
    return count / len(yhat)

def evaluate(data_dir, model_name, chkpt_path):
    assert (model_name == 'CNN' or model_name == 'CNN_LSTM' or model_name == 'CNN_Transformer'), 'Invalid Model Name'
    if model_name== 'CNN':
        model = CNNModel()
    elif model_name == 'CNN_LSTM':
        model = CNN_LSTM()
    else:
        model = CNN_Transformer()

    model.load_weights(chkpt_path)
    sgd = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    test_X = np.load(os.path.join(data_dir, 'test_X.npy'))
    test_y = np.load(os.path.join(data_dir, 'test_y.npy'))
    
    loss, acc = model.evaluate(test_X, test_y, verbose = 2)
    print("Evaluated model achieves a loss of {} and a test accuracy of {}".format(loss, acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, required = True)
    parser.add_argument("--model_name", type = str, required = True)
    parser.add_argument("--chkpt_path", type = str, required = True)
    args = parser.parse_args()

    evaluate(args.data_dir, args.model_name, args.chkpt_path)