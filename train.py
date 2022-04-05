import tensorflow as tf
from tensorflow import keras
import numpy as np

from scripts.model import Model
import scripts.dataset as dataset

# use random seed to reproduce results
np.random.seed(42)
tf.random.set_seed(42)

if __name__ == '__main__':
    preprocess_config = dataset.preprocess_config()

    train_data, val_data, test_data = dataset.load_dataset('data/cropped_full/balanced_data', target_size=(224, 224),
                                                           batch_size=64, class_mode='binary',
                                                           configuration=preprocess_config)

    model = Model()
    model.compile_model(0.0001, 0.9, 0.999, ['accuracy', 'Recall', 'Precision', 'AUC'])
    model.set_layers_trainable(False)
    history = model.train(train_data, val_data, epochs=5)
    model.set_layers_trainable(True)
    custom_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    model = model.compile_model(0.0001, 0.9, 0.999, ['accuracy', 'Recall', 'Precision', 'AUC'])
    history = model.train(train_data, val_data, 100, [custom_early_stopping])
    loss, accuracy, recall, precision, auc = model.evaluate(test_data)
    # F1 score
    f1 = 2 * ((precision * recall) / (precision + recall))

    print(f"loss: {loss}, \n"
          f"accuracy: {accuracy}, \n"
          f"recall: {recall}, \n"
          f"precision: {precision}, \n"
          f"auc: {auc}, \n"
          f"F1: {f1}")
