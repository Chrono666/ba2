import os
import random
import shutil

import cv2
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_files
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm


def preprocess_config(rotation_range=20,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      shear_range=0.15,
                      zoom_range=0.15,
                      horizontal_flip=True,
                      fill_mode='nearest'
                      ):
    """Returns a ImageDataGenerator object with preprocessing configuration

    Arguments:
        rotation_range (int): range of rotation.
        width_shift_range (float): range of width shift.
        height_shift_range (float): range of height shift.
        rescale (float): rescale factor.
        shear_range (float): range of shear.
        zoom_range (float): range of zoom.
        horizontal_flip (bool): whether to perform horizontal flip.
        fill_mode (str): fill empty space with the nearest pixel
    """
    return ImageDataGenerator(rotation_range=rotation_range,
                              width_shift_range=width_shift_range,
                              height_shift_range=height_shift_range,
                              samplewise_std_normalization=True,
                              shear_range=shear_range,
                              zoom_range=zoom_range,
                              horizontal_flip=horizontal_flip,
                              fill_mode=fill_mode
                              )


def load_dataset(path, target_size=(224, 224), batch_size=64, class_mode='binary',
                 configuration=preprocess_config()):
    """Loads a dataset from a path with the training data already preprocessed

    Arguments:
        path (str): path to the root directory of the dataset.
        target_size (tuple): size of the image after preprocessing.
        batch_size (int): size of the batch.
        class_mode (str): class mode of the dataset.
        configuration (ImageDataGenerator): ImageDataGenerator object with preprocessing configuration.
    """
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    test_path = os.path.join(path, 'test')

    train_data = configuration.flow_from_directory(train_path,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   class_mode=class_mode)
    validation_data = ImageDataGenerator(samplewise_std_normalization=True).flow_from_directory(val_path,
                                                                                                target_size=target_size,
                                                                                                batch_size=batch_size,
                                                                                                class_mode=class_mode)
    test_data = ImageDataGenerator().flow_from_directory(test_path,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         class_mode=class_mode)
    return train_data, validation_data, test_data


def load_test_set(path):
    """
    Load test set from a path with its paths and labels

    Arguments:
        path (str): path to the root directory of the dataset.
    """
    data = load_files(path)
    paths = np.array(data['filenames'])
    targets = utils.to_categorical(np.array(data['target']))
    return paths, targets


def path_to_tensor(img_path):
    """ Loads an image from a path and returns it as a tensor.

    Arguments:
        img_path (str): path to the image.
    """
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    return np.expand_dims(x, axis=0)


def paths_to_tensor(img_paths):
    """ Loads a list of images from a list of paths and returns them as a stacked list of tensors.

    Arguments:
        img_paths (list): list of paths to the images.
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def get_file_list(input_path):
    """ Returns a list of files in a directory.

    Arguments:
        input_path (str): path to the directory.
    """
    files_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # all
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                files_list.append(os.path.join(root, file))
    return files_list


def load_classify_data(input_path, image_size=(224, 224)):
    """ Loads the data used for classification.

    Arguments:
        input_path (str): path to the directory.
        image_size (tuple): height and width of the images after resizing.
    """
    files_list = get_file_list(input_path)
    tensor_img_array = []
    for filename in files_list:
        img = tf.keras.preprocessing.image.load_img(
            filename, target_size=image_size
        )
        np_img = tf.keras.preprocessing.image.img_to_array(img)
        np_img = tf.expand_dims(np_img, 0)  # Create batch axis
        tensor_img_array.append(np_img)
    return tensor_img_array, files_list


def load_images_for_grad_cam(input_path, image_size=(224, 224)):
    """ Loads the images used for Grad-CAM.

    Arguments:
        input_path (str): path to the directory.
        image_size (tuple): height and width of the images after resizing.
    """
    images_for_computing_heatmap = []
    images_for_overlay = []
    files_list = get_file_list(input_path)
    for filename in files_list:
        img = cv2.imread(filename)
        img = cv2.resize(img, image_size)
        images_for_overlay.append(img)
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=0)
        images_for_computing_heatmap.append(img)
    return images_for_computing_heatmap, [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_for_overlay]


def save_example_images(input_path, output_path, number_of_images=12):
    """Selects a random number of images from a path and saves them to a new directory

    Arguments:
        input_path (str): path to the root directory of the dataset.
        output_path (str): path to the directory where the images will be saved.
        number_of_images (int): number of images to be saved.
    """
    files_list = get_file_list(input_path)
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # all
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                files_list.append(os.path.join(root, file))

    files_to_copy = random.sample(files_list, number_of_images)

    for i, file in enumerate(files_to_copy):
        image_name = 'data_example_' + str(i) + '.jpg'
        shutil.copy(file, os.path.join(output_path, image_name))
