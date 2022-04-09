import glob
import os
import h5py
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format
from tensorflow.keras.models import load_model
from model.dataset import get_file_list, load_classify_data


def build_model(input_shape=(224, 224, 3), dropout_rate=0.25, output_activation='sigmoid'):
    model_input = tf.keras.Input(shape=input_shape)

    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=input_shape,
                                                   input_tensor=model_input)
    flat = tf.keras.layers.Flatten(name='flatten')(base_model.output)
    dense_1 = tf.keras.layers.Dense(1400)(flat)
    dropout = tf.keras.layers.Dropout(dropout_rate)(dense_1)
    batch = tf.keras.layers.BatchNormalization()(dropout)
    output = tf.keras.layers.Dense(1, activation=output_activation)(batch)
    return tf.keras.Model(base_model.input, output)


def compile_model(model, alpha, beta1, beta2, metrics):
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def set_layers_trainable(model, set_trainable):
    for layer in model.layers:
        layer.trainable = set_trainable


def train_model(model, train_data, validation_data, epochs, callbacks=[]):
    return model.fit(train_data,
                     validation_data=validation_data,
                     epochs=epochs,
                     callbacks=callbacks)


def classify_with_model(model, img_array):
    all_predictions = []
    for img in img_array:
        predictions = model.predict(img)
        all_predictions.append(predictions)
    return all_predictions


def save_model_data(model, file_path, date, model_name, dataset_name, dataset_size, train_data_size, val_data_size,
                    test_data_size):
    root_model_path = os.path.join(file_path, date)
    metadata_path = os.path.join(root_model_path, 'metadata')
    model_path = os.path.join(root_model_path, 'model')
    os.mkdir(root_model_path)
    os.mkdir(metadata_path)
    model.save(model_path)
    with h5py.File((metadata_path + '/' + model_name + '_' + date + '.hdf5'), mode='w') as f:
        hdf5_format.save_model_to_hdf5(model, f)
        f.attrs['model_name'] = model_name
        f.attrs['dataset_name'] = dataset_name
        f.attrs['data_set_size'] = dataset_size,
        f.attrs['train_data_size'] = train_data_size,
        f.attrs['val_data_size'] = val_data_size,
        f.attrs['test_data_size'] = test_data_size


def load_model_with_metadata(file_path):
    for filename in glob.glob('{}/*.hdf5'.format(file_path + '/metadata')):
        with h5py.File(filename, mode='r') as f:
            metadata = {
                'model_name': f.attrs['model_name'],
                'dataset_name': f.attrs['dataset_name'],
                'train_data_size': f.attrs['train_data_size'],
                'val_data_size': f.attrs['val_data_size'],
                'test_data_size': f.attrs['test_data_size']
            }
            model = load_model(os.path.join(file_path, 'model'))
    return model, metadata


def get_predictions_from_model(model, input_path):
    true_positives = []
    true_negatives = []
    false_positive = []
    false_negative = []
    img_arr, file_list = load_classify_data(input_path)
    for img, file in zip(img_arr, file_list):
        prediction = model.predict(img)
        if prediction[0] == 0 and file.rsplit('\\', 2)[1] == 'DEF':
            true_negatives.append((prediction, file))
            print('{} was classified correctly as DEF'.format(file.rsplit('\\', 1)[-1]))
        elif prediction[0] == 0 and file.rsplit('\\', 2)[1] == 'OK':
            false_positive.append((prediction, file))
            print('{} was classified incorrectly as OK'.format(file.rsplit('\\', 1)[-1]))
        elif prediction[0] == 1 and file.rsplit('\\', 2)[1] == 'OK':
            true_positives.append((prediction, file))
            print('{} was classified correctly as OK'.format(file.rsplit('\\', 1)[-1]))
        elif prediction[0] == 1 and file.rsplit('\\', 2)[1] == 'DEF':
            false_negative.append((prediction, file))
            print('{} was classified incorrectly as DEF'.format(file.rsplit('\\', 1)[-1]))

    return (true_positives, true_negatives), (false_positive, false_negative)



