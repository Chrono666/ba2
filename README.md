# Image Classification of infrared images taken from SLS printing processes - Practical implementation for BA2

## Table of contents

* [General information](#general-information)
* [Project information](#project-information)
* [Setup](#setup)
* [How to use the train script to train a VGG16 model](#How-to-use-the-train-script-to-train-a-VGG16-model)
* [How to use the test script to classify images with a trained VGG16 model](#How-to-use-the-test-script-to-classify-images-with-a-trained-VGG16-model)

## General information

This project is the technical implementation of the bachelor thesis "TITLE STILL OPEN FOR DEBATE" and based on
implementation done in the first bachelor thesis. A disadvantage of the layer wise manufacturing of SLS is that the
component quality can not be assured without destroying or scanning the component. For this purpose, as with the first
bachelor thesis, this thesis uses images of a layer wise SLS printing process. And uses them to train a convolutional
neural network to then later use it for image classification. In contrast to the first bachelor thesis, multiple
different geometric shapes were printed and recorded. One process without any defects occurring, and another with
increased temperature to created so-called curling defects.

## Project information

This project is based heavily on the first bachelor thesis (https://github.com/Chrono666/SLS-bachelor-project-imp).
Therefore, large parts of the code it were reused. But contrary to the first thesis, instead of jupyter notebooks,
python scripts were used. The entrance scripts are the train.py and test.py scripts. Which will start the training or
the testing of a model respectively. For training a custom VGG16 model is used, its top-layers have been removed and
exchanged with a dense sigmoid activation layer. And it is trained on two classes: curling and non-curling - OK and DEF.

## Setup

This project was created with python and anaconda. For this purpose a virtual environment was created with anaconda and
most of the packages were installed through conda, for some packages pip was used. The environment can be found in the
v_envs folder of the project (currently only windows supported).

In the command line move to the env folder of the project and use the following commands:

```bash
$ conda env create -f ba2.yaml
$ conda activate ba2
```  

or use the graphical interface (anaconda-navigator) to import it.

Also, for plotting the model architecture graphviz is used, which has to be installed separately and added to the path
variable (https://graphviz.org/download/).

#### NOTE:

This project also uses the GPU to increase computational speed and performance. What kind of device your operating
system provides is displayed by `tf.config.get_visible_devices()` and is CPU by default. To enable GPU support please
follow the instructions on https://www.tensorflow.org/install/gpu

## How to use the train script to train a VGG16 model

The train script is used to train a VGG16 model on the curling and non-curling images. For this purpose multiple
parameters can be passed to the script. All provided hyperparameters are configured with default values and therefore do
not need to be passed to the script. Important is to pass the path to the images to train on. This can be done with the
argument --data-dir. With this argument the script expects the following folder structure:

```bash
├───data
│   └───cross_geometry
│       ├───test
│       │   ├───DEF
│       │   └───OK
│       ├───train
│       │   ├───DEF
│       │   └───OK
│       └───val
│           ├───DEF
│           └───OK
```  

Naming the dataset folder in a meaningful way has the advantage that this name can later be used to associate the used
data with the trained model.

Example usage:

```bash
python train.py --data-dir cross_geometry --epochs 100 --batch-size 64 --learning-rate 0.0001 --beta-1 0.9 --beta-2 0.999
```  

With this a VGG16 custom model is trained on the images. The model is then saved once by the default keras
implementation and once including metadata in the saved_models folder. Also, a training report is generated and saved in
the target folder.

## How to use the test script to classify images with a trained VGG16 model

The test script needs the additional arguments of the location of the trained model and the location of the images to
classify. For this the following folder structure in necessary:

```bash
├───saved_models
│   ├───2022-04-08-08-54-49
│   │   ├───metadata
│   │   └───model
│   │       ├───assets
│   │       └───variables
```

If a model trained and saved through the train.py script the folder structure was automatically created by saving the
model at the end.

Currently, the folder structure for the images has to look like this, to support the report generation:

```bash
├───data_for_classification
│   ├───DEF
│   └───OK
```

Example usage:

```bash
python test.py --data-dir data_for_classification --model-dir saved_models/2022-04-08-08-54-49
```  

With this the trained VGG16 model is loaded from the model directory and used to classify the images located in the data
directory. The classification results are saved in the test report which is generated at the end of the script and can
be found in the target folder. Additionally, all generated images which are not displayed in the report are also saved in
the corresponding report folder.
