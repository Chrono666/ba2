import os

import matplotlib.pyplot as plt
import tensorflow as tf


def save_fig(name, path, tight_layout=False, fig_extension="png", resolution=300):
    path = os.path.join(path, name + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()


def plot_train_figures(history, metrics, path):
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    save_fig("loss", path)
    plt.plot(history.epoch, metrics['accuracy'], metrics['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    save_fig("accuracy", path)
    plt.plot(history.epoch, metrics['recall'], metrics['val_recall'])
    plt.legend(['recall', 'val_recall'])
    save_fig("recall", path)
    plt.plot(history.epoch, metrics['precision'], metrics['val_precision'])
    plt.legend(['precision', 'val_precision'])
    save_fig("precision", path)
    plt.plot(history.epoch, metrics['auc'], metrics['val_auc'])
    plt.legend(['auc', 'val_auc'])
    save_fig("auc", path)


def plot_model_architecture(model, file_name, path):
    file_path = os.path.join(path, file_name)
    tf.keras.utils.plot_model(model, to_file=file_path, show_shapes=True)


def plot_kernels(model, path):
    all_filters = []
    for layer in model.layers:
        if 'conv' not in layer.name:
            continue
        filters, biases = layer.get_weights()
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        all_filters.append(filters)

    n_filters, ix = len(all_filters), 1

    for i, filters in enumerate(all_filters):
        # get the filter
        f = filters[:, :, :, i]
        # plot each channel separately
        for j in range(3):
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
            save_fig(('kernel' + str(ix - 1)), path)
