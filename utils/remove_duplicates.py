import os

import cv2
from imutils import paths
from tqdm import tqdm


def dhash(image, hashSize=8):
    """ Compute the difference hash of an image.

    Arguments:
        image (str): The image(path) to compute the hash of.
        hashSize (int): The size of the hash in bits.
    """
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def remove_duplicates(input_path):
    """ Remove duplicate images from a directory by comparing their hashes.

    Arguments:
        input_path (str): The path to the directory to remove duplicates from.
    """
    imagePaths = list(paths.list_images(input_path))
    hashes = {}
    # loop over our image paths
    for imagePath in tqdm(imagePaths):
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)
        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

    # loop over the image hashes
    for (h, hashedPaths) in tqdm(hashes.items()):
        # check to see if there is more than one image with the same hash
        if len(hashedPaths) > 1:
            for p in hashedPaths[1:]:
                os.remove(p)
