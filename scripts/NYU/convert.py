#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# written by Benedikt Waldvogel

from __future__ import print_function

from joblib import Parallel, delayed
from scipy.misc import imsave
import h5py
import numpy as np
import os
import scipy.io
import sys
from PIL import Image, ImageOps

from _structure_classes import get_structure_classes
import _solarized


def processLabelImage(labelImage):
    colors = dict()
    colors["structure"] = _solarized.colors[5]
    colors["prop"] = _solarized.colors[8]
    colors["furniture"] = _solarized.colors[9]
    colors["floor"] = _solarized.colors[1]
    shape = list(labelImage.shape) + [3]
    img = np.ndarray(shape=shape, dtype=np.uint8)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            l = labelImage[i, j]
            if (l == 0):
                img[i, j] = (0, 0, 0)  # background
            else:
                name = classes[names[l - 1]]
                assert name in colors, name
                img[i, j] = colors[name]
    return img


def decodeImage(i, scene, img_depth, image, label):

    img_depth = np.asfortranarray(img_depth)

    # replace values 0.0 with NaN
    img_depth[img_depth == 0.0] = np.nan

    idx = int(i) + 1
    if idx in train_images:
        train_test = "training"
    else:
        assert idx in test_images, "index %d neither found in training set nor in test set" % idx
        train_test = "testing"

    folder = "%s/%s/%s" % (out_folder, train_test, scene)
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open("%s/%05d_depth_image.data" % (folder, i), "w") as f:
        shape = img_depth.shape
        shape = np.array(shape, dtype=np.uint16)
        shape.tofile(f)
        f.write(img_depth)

        imsave("%s/%05d_depth_image.bmp" % (folder, i), getDepthImage(img_depth))

    imsave("%s/%05d_lab_image.png" % (folder, i), image)

    labelImg = processLabelImage(label)

    imsave("%s/%05d_color_image.png" % (folder, i), labelImg)


def getDepthImage(image):
    width, height = image.shape
    b = np.frombuffer(image.data, dtype=np.float32).copy()

    maxdepth = np.nanmax(b) * 1.0
    b /= maxdepth
    b *= 256.0
    b = 256 - b
    b = np.nan_to_num(b)
    b = np.maximum(b, 0)
    b = b.astype(np.uint8)
    size = (width, height)
    img = Image.frombuffer("L", size, b, "raw", "L", 0, 1)
    img = ImageOps.equalize(img)
    return img


if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("usage: %s <h5_file> <train_test_split> <out_folder> [<rawDepth> <num_threads>]" % sys.argv[0], file=sys.stderr)
        sys.exit(0)

    h5_file = h5py.File(sys.argv[1], "r")
    # h5py is not able to open that file. but scipy is
    train_test = scipy.io.loadmat(sys.argv[2])
    out_folder = sys.argv[3]
    if len(sys.argv) >= 5:
        raw_depth = bool(int(sys.argv[4]))
    else:
        raw_depth = False

    if len(sys.argv) >= 6:
        num_threads = int(sys.argv[5])
    else:
        num_threads = -1

    test_images = set([int(x) for x in train_test["testNdxs"]])
    train_images = set([int(x) for x in train_test["trainNdxs"]])
    print("%d training images" % len(train_images))
    print("%d test images" % len(test_images))

    if raw_depth:
        print("using raw depth images")
        depth = h5_file['rawDepths']
    else:
        print("using impainted depth images")
        depth = h5_file['depths']

    print("reading", sys.argv[1])

    labels = h5_file['labels']
    images = h5_file['images']

    rawDepthFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawDepthFilenames'][0]]
    names = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['names'][0]]
    scenes = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['sceneTypes'][0]]
    rawRgbFilenames = [u''.join(unichr(c) for c in h5_file[obj_ref]) for obj_ref in h5_file['rawRgbFilenames'][0]]
    classes = get_structure_classes()

    print("processing images")
    if num_threads == 1:
        print("single-threaded mode")
        for i, image in enumerate(images):
            print("image", i + 1, "/", len(images))
            decodeImage(i, scenes[i], depth[i, :, :], image.T, labels[i, :, :].T)
    else:
        Parallel(num_threads, 5)(delayed(decodeImage)(i, scenes[i], depth[i, :, :], images[i, :, :].T, labels[i, :, :].T) for i in range(len(images)))

    print("finished")
