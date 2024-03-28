# EEG Data Sleep Classifer
Our research project aimed to develop and train various classifiers to uncover the current sleep stage of a 30-second epoch from a raw single-channel EEG signal. We developed three separate models and compared the results to each other. 

## Reproducing Results
For ease of reproduction, we have created a jupyter notebook called `ResultReproductionNotebook.ipynb` which contains not only code for the entire pipeline but also explanations along the way of choices we made. Furthermore, the repo above contains more modularized classes of the separate models as well as various scripts for each individual aspect of the data lifecycle. 

## Available Scripts
* Preprocess.py
  * Args required
    * --data_dir: location to .npz files containing eeg data:
    * --output_dir: location to save test and train data
  * Output: This script will accept an input location for data and save to output preprocessed train and test sets as train_X.npy, train_y.npy, test_X.npy, and test_y.npy. For reference we utilized the fpz_cz channel data located in 'data/eeg_fpz_cz'
* HyperparameterTuning.py
  * Args Required
    * --data_dir: location to train .npy files
    * --model_name: name of model to tune
  * Output: This script accepts the location to the train files outputted from Preprocess.py as well as a model name parameter that must be either 'CNN' or 'CNN_LSTM' to perform hyperparameter tuning. After training it will output the best parameter configuration that it found.
* train.py
  * Args Required
    * --data_dir: location to train .npy files
    * --chkpt_dir: location to save model checkpoints
    * --model_name: name of model to train
  * Output: This script accepts the location of the train files outputted from Preprocess.py, the location where to save the model checkpoints, and a model name parameter that must be either 'CNN', 'CNN_LSTM', or 'CNN_Transformer'
* EvaluateModel.py
  * Args Required
    * --data_dir: location to test .npy files
    * --model_name: name of model to traini
    * --chkpt_path: path to specific model checkpoint to use
  * Output: This script accepts the location to the test files outputted from Preprocess.py, a model name parameter that must be either 'CNN', 'CNN_LSTM', or 'CNN_Transformer', and a path to a checkpoint outputted from train.py and returns the test loss and accuracy of the model.

## Environment
This code was tested and ran using:
* Windows 10.0.19043
* Numpy version 1.22.0
* Pandas version 1.4.0
* TensorFlow version 2.8.0
* Python version 3.10.4
