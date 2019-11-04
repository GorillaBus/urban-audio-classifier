import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time


"""
    Data processing
"""


# Generates/extracts SFTF-MFCC coefficients with Librosa 
def get_sftf_mfcc(filename, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        y, sr = librosa.load(filename)
        normalized_y = librosa.util.normalize(y)
        
        stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)
        mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)
        mellog = np.log(mel + 1e-9)
        melnormalized = librosa.util.normalize(mellog)

        # Should we require padding
        shape = melnormalized.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            melnormalized = np.pad(melnormalized, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", filename)
        return None 
    return melnormalized


# Generates/extracts MFCC coefficients with Librosa 
def get_mfcc(filename, mfcc_max_padding=0, n_mfcc=128):
    try:
        audio, sample_rate = librosa.load(filename) 
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Should we require padding
        shape = mfcc.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            mfcc = np.pad(mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", filename)
        return None 
    return mfcc


# Scales data between x_min and x_max
def scale(X, x_min, x_max, axis=0):
    nom = (X-X.min(axis=axis))*(x_max-x_min)
    denom = X.max(axis=axis) - X.min(axis=axis)
    denom[denom==0] = 1
    return x_min + nom/denom 


def save_split_distributions(test_split_idx, train_split_idx, file_path=None):
    if (path == None):
        print("You must enter a file path to save the splits")        
        return false

    
    # Create split dictionary
    split = {}
    split['test_split_idx'] = test_split_idx
    split['train_split_idx'] = train_split_idx

    with open(file_path, 'wb') as file_pi:
        pickle.dump(split, file_pi)

    return file


def load_split_distributions(file_name, path='./data'):
    file_path = os.path.join(path, file_name)

    file = open(file_path, 'rb')
    data = pickle.load(file)
    
    return [data['test_split_idx'], data['train_split_idx']]
  


"""
    Prediction and analisys
"""

def evaluate_model(model, X_train, y_train, X_test, y_test):
    dash = '-' * 38
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)

    # Calculate the error gap between Train and Test loss
    a = max(train_score[0], test_score[0])
    b = min(train_score[0], test_score[0])
    loss_diff = a - b
    loss_gap = loss_diff * 100 / a

    # Pint Train vs Test results
    print('{:<10s}{:>14s}{:>14s}'.format( "", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))
    print('{:<10s}{:>13.2f}{:>14s}'.format( "Error gap %", loss_gap, ""))

    return


# Expects a NumPy array with probabilities and a confusion matrix data, retuns accuracy per class
def acc_per_class(np_probs_array):    
    accs = []
    for idx in range(0, np_probs_array.shape[0]):
        correct = np_probs_array[idx][idx].astype(int)
        total = np_probs_array[idx].sum().astype(int)
        acc = (correct / total) * 100
        accs.append(acc)
    return accs




"""
    Plotting
"""

def plot_train_history(history):
    history = history.history

    # min loss / max accs
    min_loss = min(history['loss'])
    min_val_loss = min(history['val_loss'])
    max_accuracy = max(history['accuracy'])
    max_val_accuracy = max(history['val_accuracy'])

    # x pos for loss / acc min/max
    min_loss_x = history['loss'].index(min_loss)
    min_val_loss_x = history['val_loss'].index(min_val_loss)
    max_accuracy_x = history['accuracy'].index(max_accuracy)
    max_val_accuracy_x = history['val_accuracy'].index(max_val_accuracy)

    # summarize history for loss, display min
    plt.figure(figsize=(16,8))
    plt.plot(history['loss'], color="#1f77b4", alpha=0.7)
    plt.plot(history['val_loss'], color="#ff7f0e", linestyle="--")
    plt.plot(min_loss_x, min_loss, marker='o', markersize=3, color="#1f77b4", alpha=0.7, label='Inline label')
    plt.plot(min_val_loss_x, min_val_loss, marker='o', markersize=3, color="#ff7f0e", alpha=7, label='Inline label')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.3f' % min_loss), 
                ('%.3f' % min_val_loss)], 
                loc='upper right', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)
    plt.xticks(np.arange(0, len(history['loss']), 5.0))
    plt.show()



    # summarize history for accuracy, display max
    plt.figure(figsize=(16,6))
    plt.plot(history['accuracy'], alpha=0.7)
    plt.plot(history['val_accuracy'], linestyle="--")
    plt.plot(max_accuracy_x, max_accuracy, marker='o', markersize=3, color="#1f77b4", alpha=7)
    plt.plot(max_val_accuracy_x, max_val_accuracy, marker='o', markersize=3, color="orange", alpha=7)
    plt.title('Model accuracy', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=16)
    plt.legend(['Train', 
                'Test', 
                ('%.2f' % max_accuracy), 
                ('%.2f' % max_val_accuracy)], 
                loc='upper left', 
                fancybox=True, 
                framealpha=0.9, 
                shadow=True, 
                borderpad=1)
    plt.figure(num=1, figsize=(10, 6))
    plt.xticks(np.arange(0, len(history['accuracy']), 5.0))
    plt.show()


