import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_config(rotation_range=20,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      rescale=1. / 255,
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
                              rescale=rescale,
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
    validation_data = ImageDataGenerator().flow_from_directory(val_path,
                                                               target_size=target_size,
                                                               batch_size=batch_size,
                                                               class_mode=class_mode)
    test_data = ImageDataGenerator().flow_from_directory(test_path,
                                                         target_size=target_size,
                                                         batch_size=batch_size,
                                                         class_mode=class_mode)
    return train_data, validation_data, test_data
