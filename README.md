# Urban sounds classification with Convolutional Neural Networks

The objective of this project is to implement CNN models to recognize sound events from the **UrbanSound9K** dataset. The work has been divided into the following notebooks:

1. Data analysis (and related papers brief)
2. Pre-processing and feature evaluation
3. CNN model with MFCC 
4. CNN model with Log-MEL Spectrograms
5. Data augmentation
6. Data augmentation pre-processing
7. CNN model with augmented data (Log-MEL Spectrograms)

## Notebooks

1. [Data analysis](https://github.com/GorillaBus/urban-audio-classifier/blob/master/1-data-analysis.ipynb): a brief about previous works with the URbanSound8K dataset (scientific papers), dataset exploration, distribution analysis, listening.

2. [Pre-processing](https://github.com/GorillaBus/urban-audio-classifier/blob/master/2-pre-processing.ipynb): an introduction to different audible features we can use to work with digital audio, the pre-processing pipeline, STFT, MFCC and Log-MEL Spectrograms, feature extraction and data normalization.

3. [CNN model with MFCC features](https://github.com/GorillaBus/urban-audio-classifier/blob/master/3-cnn-model-mfcc.ipynb): data preparation, CNN model definition (with detailed explanation) using Keras and TensorFlow back-end. Solution of a multi-class classification problem, model evaluation and testing, Recall, Precision and F1 analysis.

4. [CNN Model with Log-MEL Spectrograms](https://github.com/GorillaBus/urban-audio-classifier/blob/master/4-cnn-model-mel_spec.ipynb): a performance comparison using the same CNN model architecture with MEL spectrograms. Same training and evaluation than notebook #3.

5. [Data augmentation](https://github.com/GorillaBus/urban-audio-classifier/blob/master/5-data-augmentation.ipynb): creation of augmented data from UrbanSound8K original sounds, using common audio effects like pitch shifting, time stretching, adding noise, with LibROSA.

6. [Augmented pre-processing](https://github.com/GorillaBus/urban-audio-classifier/blob/master/6-augmented-pre-processing.ipynb): audible features extraction from the new generated data.

7. [CNN model with augmented data](https://github.com/GorillaBus/urban-audio-classifier/blob/master/7-cnn-model-augmented.ipynb): using the same CNN architecture and almost identical training procedures with the generated data. Model evaluation and test to compare with previous achievements.


## Getting the dataset

Download a copy of the UrbanSounds8K dataset from the [UrbanSound8K home page](https://urbansounddataset.weebly.com/urbansound8k.html).

Make sure to uncompress the dataset root directory into the project root, you should end up with a directory like "UrbanSounds8K" (or a symbolic link to it) in the project root.


## Install required libraries

Make sure that Tensorflow, Keras, LibROSA, IPython, NumPy, Pandas, Matplotlib and SciKit Learn are already installed in your environment.

Note that we are using Tensorflow as Keras back-end, you must set this in your ~/.keras/keras.json file, this is an example:

```
{
    "image_dim_ordering": "tf",
    "image_data_format": "channels_first",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

## The UrbanSound8K dataset

The **UrbanSound8K** dataset is a compilation of urban sound recordings, classified in 10 categories according to the paper ["A Dataset and Taxonomy for Urban Sound Research"](https://urbansounddataset.weebly.com/taxonomy.html), which proposes a taxonomical categorization to describe different environmental sound types.

The UrbanSound8K dataset contains 8732 labeled sound slices of varying duration up to 4 seconds. The categorization labels being:

1. Air Conditioner
1. Car Horn
1. Children Playing
1. Dog bark
1. Drilling
1. Engine Idling
1. Gun Shot
1. Jackhammer
1. Siren
1. Street Music

Note that the dataset comes already organized in 10 validation folds. In the case we want to compare our results with other we should stick with this schema.


### Dataset metadata

The included metadata file ("UrbanSound8K/metadata/metadata.csv") provides all the required information about each audio file:

* slice_file_name: The name of the audio file.
* fsID: The Freesound ID of the recording from which this excerpt (slice) is taken
* start: The start time of the slice in the original Freesound recording
* end: The end time of slice in the original Freesound recording
* salience: A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
* fold: The fold number (1-10) to which this file has been allocated.
* classID: A numeric identifier of the sound class.
* class: The class label name.



## References

1- Data analysis
* Taxonomical categorization (resume): https://urbansounddataset.weebly.com/taxonomy.html
* "A Dataset and Taxonomy for Urban Sound Research":
http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf
* Basics of Digital Audio: https://theproaudiofiles.com/digital-audio-101-the-basics/
* Reading wave file headers with Python: https://www.cameronmacleod.com/blog/reading-wave-python
* The Wave PCM file specification: http://soundfile.sapp.org/doc/WaveFormat/ 

2- Data pre-processing
* The Nyquist theorem: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
* Sampling : Signal Digitalization: http://support.ircam.fr/docs/AudioSculpt/3.0/co/Sampling.html
* Digital audio conversion: what is Aliasing? https://theproaudiofiles.com/digital-audio-aliasing/
* Mel Frequency Cepstral Coefficient (MFCC) tutorial: http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
* Discussion on odd kernel sizes: https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
* Discussion about normalized audio for CNNs: https://stackoverflow.com/questions/55513652/which-spectrogram-best-represents-features-of-an-audio-file-for-cnn-based-model/56727927#56727927
* A Comparison of Audio Signal Preprocessing Methods for Deep Neural Networks on Music Tagging: https://arxiv.org/abs/1709.01922

4- Model optimization
* Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
* CLR Keras implementation by Brad Kenstler: https://github.com/bckenstler/CLR

5- Related papers
* Environmental sound classification with convolutional neural networks, Karol J. Piczak
* Dilated convolution neural network with LeakyReLU for environmental sound classification, Xiaohu Zhang ; Yuexian Zou ; Wei Shi.

* Deep Convolutional Neural Network with Mixup for Environmental Sound Classification, Zhichao Zhang, Shugong Xu, Shan Cao, Shunqing Zhang

* End-to-End Environmental Sound Classification using a 1DConvolutional Neural NetworkSajjad Abdoli, Patrick Cardinal, Alessandro Lameiras Koerich

* An Ensemble Stacked Convolutional Neural Network Model for Environmental Event Sound Recognition, Shaobo Li, Yong Yao, Jie Hu, Guokai Liu, Xuemei Yao 3, Jianjun Hu

* Classifying environmental sounds using image recognition networks, Venkatesh Boddapati, Andrej Petef, Jim Rasmusson, Lars Lundberg

* Environment Sound Classification Using a Two-Stream CNN Based on Decision-Level Fusion, Yu Su, Ke Zhang, Jingyu Wang, Kurosh Madani


### Comments, suggestions and corrections always welcome 
