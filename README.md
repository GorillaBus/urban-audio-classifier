# Urban sounds classification with Covnolutional Neural Networks

### *** Under development ***

The objective of this project is to implement CNN models to correctly classify the sounds of the UrbanSound9K dataset. The work has been divided in:

* Data analysis
* Pre-processing
* CNN Model definition 
* Model training and validation
* Model optimization
* Data augmentation
* Conclusions


## 1. Requirements

### Getting the dataset

Download a copy of the UrbanSounds8K dataset from the [UrbanSound8K home page](https://urbansounddataset.weebly.com/urbansound8k.html).

Make sure to uncompress the dataset root directory into the project root, you should end up with a directory like "UrbanSounds8K" (or a symbolic link to it) in the project root.


### 2. Install required libraries

Make sure that Tensorflow, Keras, LibROSA, IPython, NumPy, Pandas, Matplotlib and SciKit Learn are already installed in your system / virtual environment.

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

## 2. The UrbanSound8K dataset

The "UrbanSound8K" dataset is a compilation of urban sound recordings, classified in 10 categories according to the paper ["A Dataset and Taxonomy for Urban Sound Research"](https://urbansounddataset.weebly.com/taxonomy.html), which proposes a taxonomical categorization to describe different environmental sound types.

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


#### Dataset metadata

The included metadata file ("UrbanSound8K/metadata/metadata.csv") provides all the required information about each audio file:

* slice_file_name: The name of the audio file.
* fsID: The Freesound ID of the recording from which this excerpt (slice) is taken
* start: The start time of the slice in the original Freesound recording
* end: The end time of slice in the original Freesound recording
* salience: A (subjective) salience rating of the sound. 1 = foreground, 2 = background.
* fold: The fold number (1-10) to which this file has been allocated.
* classID: A numeric identifier of the sound class.
* class: The class label name.



## N. References

1- Data analisys
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
