import glob
import os
from pathlib import PurePath

import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.saving import hdf5_format
from tqdm import tqdm


def build_model(input_shape=(224, 224, 3), dropout_rate=0.25, output_activation='sigmoid'):
    """ Builds the custom VGG16 model by loading a pre-trained model and adding a custom fully connected layer.

    Arguments:
        input_shape {tuple} -- The shape of the input data.
        dropout_rate {float} -- The dropout rate to use.
        output_activation {str} -- The activation function to use for the output layer.
    """
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
    """ Compiles the model with the given hyperparameters.

    Arguments:
        model {tf.keras.Model} -- The model to compile.
        alpha {float} -- The learning rate.
        beta1 {float} -- The beta1 hyperparameter.
        beta2 {float} -- The beta2 hyperparameter.
        metrics {list} -- The metrics that will be watched during training.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
    return model


def set_layers_trainable(model, set_trainable, layer_number):
    """ Sets the trainable property of all layers in the model.

    Arguments:
        model {tf.keras.Model} -- The model to modify.
        set_trainable {bool} -- Whether the layers should be trainable.
        layer_number {int} -- Number of layers to be frozen.
    """
    for index, layer in enumerate(model.layers):
        layer.trainable = set_trainable
        if index == layer_number:
            break


def set_base_model_layers_trainable(model, set_trainable):
    """ Sets the trainable property of all layers in the BASE model.

    Arguments:
        model {tf.keras.Model} -- The model to modify.
        set_trainable {bool} -- Whether the layers should be trainable.
    """
    for layer in model.layers:
        if layer.name == 'flatten':
            break
        layer.trainable = set_trainable
    for layer in model.layers:
        print(layer.name, layer.trainable)


def set_conv_layers_trainable(model, set_trainable, conv_layer_number):
    """ Sets the trainable property of conv layers in the model.

    Arguments:
        model {tf.keras.Model} -- The model to modify.
        set_trainable {bool} -- Whether the layers should be trainable.
        conv_layer_number {int} -- Number of conv layers to be set.
    """
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name]
    for layer in conv_layers[conv_layer_number:]:
        layer.trainable = set_trainable
    for layer in model.layers:
        print(layer.name, layer.trainable)
    print('Trainable layers: ' + str(len(conv_layers[conv_layer_number:])) + '/' + str(len(conv_layers)))


def train_model(model, train_data, validation_data, epochs, callbacks=[]):
    """ Trains the model with the given data.

    Arguments:
        model {tf.keras.Model} -- The model to train.
        train_data {tuple} -- The training data.
        validation_data {tuple} -- The validation data.
        epochs {int} -- The number of epochs to train for.
        callbacks {list} -- The callbacks to use during training.
    """
    return model.fit(train_data,
                     validation_data=validation_data,
                     epochs=epochs,
                     callbacks=callbacks)


def classify_with_model(model, img_array):
    """ Classifies the given image array with the model.

    Arguments:
        model {tf.keras.Model} -- The model to use for classification.
        img_array {np.ndarray} -- The image array to classify.
    """
    all_predictions = []
    for img in img_array:
        predictions = model.predict(img)
        all_predictions.append(predictions)
    return all_predictions


def save_model_data(model, file_path, date, model_name, dataset_name, dataset_size, train_data_size, val_data_size,
                    test_data_size):
    """ Saves the model once by using the keras model.save method.
        and once in h5 format containing metadata used for reporting.

    Arguments:
        model {tf.keras.Model} -- The model to save.
        file_path {str} -- The path to save the model to.
        date {str} -- The date the model was saved.
        model_name {str} -- The name of the model.
        dataset_name {str} -- The name of the dataset.
        dataset_size {int} -- The size of the dataset.
        train_data_size {int} -- The size of the training data.
        val_data_size {int} -- The size of the validation data.
        test_data_size {int} -- The size of the test data.
    """
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
    """ Loads a model and its metadata from the given file path.

    Arguments:
        file_path {str} -- The path to the model and metadata.
    """
    for filename in glob.glob('{}/*.hdf5'.format(file_path + '/metadata')):
        with h5py.File(filename, mode='r') as f:
            metadata = {
                'model_name': f.attrs['model_name'],
                'dataset_name': f.attrs['dataset_name'],
                'dataset_size': f.attrs['data_set_size'],
                'train_data_size': f.attrs['train_data_size'],
                'val_data_size': f.attrs['val_data_size'],
                'test_data_size': f.attrs['test_data_size']
            }
    model = load_model(os.path.join(file_path, 'model'))
    return model, metadata


def load_old_ba1_model(model_path, weight_path):
    """ Load BA1 trained model.

    Arguments:
        model_path {str} -- Path to the model.
        weight_path {str} -- Path to the weights.
    """
    # load json and create model
    file = open(model_path, 'r')
    model_json = file.read()
    file.close()
    loaded_model = tf.keras.models.model_from_json(model_json)
    # load weights
    loaded_model.load_weights(weight_path)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    loaded_model.compile(loss="binary_crossentropy", optimizer=optimizer,
                         metrics=['accuracy', 'Recall', 'Precision', 'AUC'])
    return loaded_model


def get_predictions_from_model(model, img_arr, file_list):
    """ Used a model to classify images and sort them according
        to true positives, true negatives, false positives, false negatives.

    Arguments:
        model {tf.keras.Model} -- The model to use for classification.
        img_arr {np.ndarray} -- The image array to classify.
        file_list {list} -- file path of images to reconstruct TP, TN, FP, FN.
    """
    true_positives = []
    true_negatives = []
    false_positive = []
    false_negative = []
    for img, file in tqdm(zip(img_arr, file_list)):
        prediction = model.predict(img)
        path_parts = PurePath(file).parts
        if prediction[0] == 0 and path_parts[-2] == 'DEF':
            true_negatives.append((prediction, file))
            # print('{} was classified correctly as DEF'.format(path_parts[-2]))
        elif prediction[0] == 0 and path_parts[-2] == 'OK':
            false_negative.append((prediction, file))
            # print('{} was classified incorrectly as OK'.format(path_parts[-2]))
        elif prediction[0] == 1 and path_parts[-2] == 'OK':
            true_positives.append((prediction, file))
            # print('{} was classified correctly as OK'.format(path_parts[-2]))
        elif prediction[0] == 1 and path_parts[-2] == 'DEF':
            false_positive.append((prediction, file))
            # print('{} was classified incorrectly as DEF'.format(path_parts[-2]))
    return true_positives, true_negatives, false_positive, false_negative
