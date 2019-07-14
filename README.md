# Urban sounds classification with Covnolutional Neural Networks

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



## N. References

* Taxonomical categorization (resume): https://urbansounddataset.weebly.com/taxonomy.html
* "A Dataset and Taxonomy for Urban Sound Research":
http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf
* If you are new to Digital Audio: https://theproaudiofiles.com/digital-audio-101-the-basics/
* Digital audio conversion: what is Aliasing? https://theproaudiofiles.com/digital-audio-aliasing/
