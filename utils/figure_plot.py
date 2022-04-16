import os
import random

from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from grad_cam.grad_cam import GradCAM


def save_fig(name, path, tight_layout=False, fig_extension="png", resolution=300):
    """Save figure to filesystem.

    Args:
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


def plot_grad_cams(model, images_for_heatmap, images, path, conv_layer_name, ):
    cam = GradCAM(model, conv_layer_name)
    heatmaps = [cam.compute_heatmap(img) for img in images_for_heatmap]
    counter = 0
    for a, i in tqdm(zip(heatmaps, images)):
        test, output = GradCAM.overlay_heatmap(heatmap=a, image=i, alpha=0.5)
        plt.imshow(output)
        save_fig(('grad_cam' + str(counter)), path)
        counter += 1


def extract_feature_maps_from_conv_layers(model, images, path):
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
    for index, img_path in enumerate(img_paths):
        img = plt.imread(img_path)
        plt.imshow(img)
        save_fig((img_prefix + str(index)), output_path)
