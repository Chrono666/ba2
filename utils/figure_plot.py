import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

from grad_cam.grad_cam import GradCAM
from model.dataset import load_images_for_grad_cam


def save_fig(name, path, tight_layout=False, fig_extension="png", resolution=300):
    """Save figure to filesystem.

    Arguments:
        name (str): Name of figure.
        path (str): Root directory.
        fig_extension (str): File extension.
        resolution (int): Resolution of figure.
        tight_layout (bool): Whether to use tight layout.
    """
    path = os.path.join(path, name + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.close()


def plot_train_figures(history, path):
    """ Plot and save the comparison graphs of the training metrics.

    Arguments:
        history (keras.model): History object returned by keras model.fit.
        path (str): Root directory.
    """
    plt.plot(history.epoch, history.history['loss'], history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    save_fig("loss", path)
    plt.plot(history.epoch, history.history['accuracy'], history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    save_fig("accuracy", path)
    plt.plot(history.epoch, history.history['recall'], history.history['val_recall'])
    plt.legend(['recall', 'val_recall'])
    save_fig("recall", path)
    plt.plot(history.epoch, history.history['precision'], history.history['val_precision'])
    plt.legend(['precision', 'val_precision'])
    save_fig("precision", path)
    plt.plot(history.epoch, history.history['auc'], history.history['val_auc'])
    plt.legend(['auc', 'val_auc'])
    save_fig("auc", path)


def plot_model_architecture(model, file_name, path):
    """Plot and save the model architecture.

    Arguments:
        model (keras.model): Model object.
        file_name (str): Name of the file.
        path (str): Path where to save the image.
    """
    file_path = os.path.join(path, file_name)
    tf.keras.utils.plot_model(model, to_file=file_path, show_shapes=True)


def plot_kernels(model, path):
    """ Extracts the kernels from the model and saves them as images.

    Arguments:
        model (keras.model): Model object.
        path (str): Path where to save the images.
    """
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


def plot_grad_cams(model, file_list, path, conv_layer_name, ):
    """ Uses the GradCAM algorithm to generate heatmaps of the images.

    Arguments:
        model (keras.model): Model object.
        path (str): Path where to save the images.
        file_list (list): List of image paths.
        conv_layer_name (str): Name of the last convolutional layer.
    """
    img_to_compute_hm, images_for_hm_overlay = load_images_for_grad_cam(file_list)
    cam = GradCAM(model, conv_layer_name)
    heatmaps = [cam.compute_heatmap(img) for img in img_to_compute_hm]
    counter = 0
    for a, i in tqdm(zip(heatmaps, images_for_hm_overlay)):
        test, output = GradCAM.overlay_heatmap(heatmap=a, image=i, alpha=0.5)
        plt.imshow(output)
        save_fig(('grad_cam' + str(counter)), path)
        counter += 1


def extract_feature_maps_from_conv_layers(model, images, path):
    """ Extracts the feature maps from the convolutional layers.

    Arguments:
        model (keras.model): Model object.
        images (list): List of images where a random one is chosen to generate the feature maps.
        path (str): Path where to save the images.
    """
    img = random.choice(images)
    for layer in tqdm(model.layers):
        if 'conv' not in layer.name:
            continue
        else:
            model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            feature_maps = model.predict(img)
            # plot all 64 maps in an 8x8 squares
            square = 8
            ix = 1
            for _ in range(square):
                for _ in range(square):
                    # specify subplot and turn of axis
                    ax = plt.subplot(square, square, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
                    ix += 1
            save_fig(layer.name, path)


def plot_classified_images(img_paths, output_path, img_prefix):
    """ Plots and saves the images with the predicted class.

    Arguments:
        img_paths (list): List of paths to the images.
        output_path (str): Path where to save the images.
        img_prefix (str): Prefix of the images.
    """
    for index, img_path in tqdm(enumerate(img_paths)):
        img = plt.imread(img_path)
        plt.imshow(img)
        save_fig((img_prefix + str(index)), output_path)
