import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.saving import hdf5_format
import h5py


class Model:
    def __init__(self, input_shape=(224, 224, 3), dropout_rate=0.25, output_activation='sigmoid'):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.model = self.__build_model()

    def __build_model(self):
        model_input = tf.keras.Input(shape=self.input_shape)

        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape,
                           input_tensor=model_input)

        flat = tf.keras.layers.Flatten(name='flatten')(base_model.output)
        dense_1 = tf.keras.layers.Dense(1400)(flat)
        dropout = tf.keras.layers.Dropout(self.dropout_rate)(dense_1)
        batch = tf.keras.layers.BatchNormalization()(dropout)
        output = tf.keras.layers.Dense(1, activation=self.output_activation)(batch)
        return tf.keras.Model(base_model.input, output)

    def compile_model(self, learning_rate, beta_1, beta_2, metrics_to_watch):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
        return self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics_to_watch)

    def set_layers_trainable(self, set_trainable):
        for layer in self.model.layers:
            layer.trainable = set_trainable

    def train(self, train_data, validation_data, epochs, callbacks=[]):
        return self.model.fit(train_data,
                              validation_data=validation_data,
                              epochs=epochs,
                              callbacks=callbacks)

    def save_model(self, file_path, *args):
        with h5py.File(file_path, mode='w') as f:
            hdf5_format.save_model_to_hdf5(self.model, f)
            f.attrs['param1'] = args[0]
            f.attrs['param2'] = args[1]

    def load_model(self, file_path):
        # Load model
        with h5py.File(model_path, mode='r') as f:
            param1 = f.attrs['param1']
            param2 = f.attrs['param2']
            my_keras_model = hdf5_format.load_model_from_hdf5(f)
