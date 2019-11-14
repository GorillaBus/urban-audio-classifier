import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import metrics 
import os
import pickle
import time
import struct




""" 
    Data processing
"""


# Generates/extracts Log-MEL Spectrogram coefficients with LibRosa 
def get_mel_spectrogram(file_path, mfcc_max_padding=0, n_fft=2048, hop_length=512, n_mels=128):
    try:
        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Generate mel scaled filterbanks
        mel = librosa.feature.melspectrogram(normalized_y, sr=sr, n_mels=n_mels)

        # Convert sound intensity to log amplitude:
        mel_db = librosa.amplitude_to_db(abs(mel))

        # Normalize between -1 and 1
        normalized_mel = librosa.util.normalize(mel_db)

        # Should we require padding
        shape = normalized_mel.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mel = np.pad(normalized_mel, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mel


# Generates/extracts MFCC coefficients with LibRosa 
def get_mfcc(file_path, mfcc_max_padding=0, n_mfcc=40):
    try:
        # Load audio file
        y, sr = librosa.load(file_path)

        # Normalize audio data between -1 and 1
        normalized_y = librosa.util.normalize(y)

        # Compute MFCC coefficients
        mfcc = librosa.feature.mfcc(y=normalized_y, sr=sr, n_mfcc=n_mfcc)

        # Normalize MFCC between -1 and 1
        normalized_mfcc = librosa.util.normalize(mfcc)

        # Should we require padding
        shape = normalized_mfcc.shape[1]
        if (mfcc_max_padding > 0 & shape < mfcc_max_padding):
            xDiff = mfcc_max_padding - shape
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            normalized_mfcc = np.pad(normalized_mfcc, pad_width=((0,0), (xLeft, xRight)), mode='constant')

    except Exception as e:
        print("Error parsing wavefile: ", e)
        return None 
    return normalized_mfcc


# Given an numpy array of features, zero-pads each ocurrence to max_padding
def add_padding(features, mfcc_max_padding=174):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px[0])
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((0,0), (xLeft, xRight)), mode='constant')
        
        padded.append(px)

    return padded


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


def load_split_distributions(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    return [data['test_split_idx'], data['train_split_idx']]
  

def find_dupes(array):
    seen = {}
    dupes = []

    for x in array:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return len(dupes)


# Reads a file's header data and returns a list of wavefile properties
def read_header(filename):
    wave = open(filename,"rb")
    riff = wave.read(12)
    fmat = wave.read(36)
    num_channels_string = fmat[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]
    sample_rate_string = fmat[12:16]
    sample_rate = struct.unpack("<I",sample_rate_string)[0]
    bit_depth_string = fmat[22:24]
    bit_depth = struct.unpack("<H",bit_depth_string)[0]
    return (num_channels, sample_rate, bit_depth)


# Given a dataset row it returns an audio player and prints the audio properties
def play_dataset_sample(dataset_row, audio_path):
    fold_num = dataset_row.iloc[0]['fold']
    file_name = dataset_row.iloc[0]['file']
    file_path = os.path.join(audio_path, fold_num, file_name)
    file_path = os.path.join(audio_path, dataset_row.iloc[0]['fold'], dataset_row.iloc[0]['file'])

    print("Class:", dataset_row.iloc[0]['class'])
    print("File:", file_path)
    print("Sample rate:", dataset_row.iloc[0]['sample_rate'])
    print("Bit depth:", dataset_row.iloc[0]['bit_depth'])
    print("Duration {} seconds".format(dataset_row.iloc[0]['duration']))
    
    # Sound preview
    return IP.display.Audio(file_path)



"""
    Prediction and analisys
"""

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_score = model.evaluate(X_train, y_train, verbose=0)
    test_score = model.evaluate(X_test, y_test, verbose=0)
    return train_score, test_score


def model_evaluation_report(model, X_train, y_train, X_test, y_test, calc_normal=True):
    dash = '-' * 38

    # Compute scores
    train_score, test_score = evaluate_model(model, X_train, y_train, X_test, y_test)

    # Pint Train vs Test report
    print('{:<10s}{:>14s}{:>14s}'.format("", "LOSS", "ACCURACY"))
    print(dash)
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Training:", train_score[0], 100 * train_score[1]))
    print('{:<10s}{:>14.4f}{:>14.4f}'.format( "Test:", test_score[0], 100 * test_score[1]))


    # Calculate and report normalized error difference?
    if (calc_normal):
        max_err = max(train_score[0], test_score[0])
        error_diff = max_err - min(train_score[0], test_score[0])
        normal_diff = error_diff * 100 / max_err
        print('{:<10s}{:>13.2f}{:>1s}'.format("Normal diff ", normal_diff, ""))



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

def plot_train_history(history, x_ticks_vertical=False):
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

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['loss']), 5.0), rotation='vertical')
    else:
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

    if (x_ticks_vertical):
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0), rotation='vertical')
    else:
        plt.xticks(np.arange(0, len(history['accuracy']), 5.0))

    plt.show()


def compute_confusion_matrix(y_true, 
               y_pred, 
               classes, 
               normalize=False):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return cm


# Plots a confussion matrix
def plot_confusion_matrix(cm,
                          classes, 
                          normalized=False, 
                          title=None, 
                          cmap=plt.cm.Blues,
                          size=(10,10)):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalized else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.show()