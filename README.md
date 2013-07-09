CUDA Random Forests for Image Labeling (CURFIL)
-----------------------------------------------

This project is an open source implementation with NVIDIA CUDA™ that accelerates random
forest training and prediction for image labeling by using the
massive parallel computing power offered by GPUs.

Implemented Visual Features
---------------------------

We currently focus on image labelling tasks such as image segmentation or classification applications.
We implement two types of RGB-D image features.

For a given query pixel, the image feature is calculated as the difference of two rectangular offset
regions on the image channel in the neighbourhood around the query pixel.
Extent size and relative offset of the rectangular region in the image is normalized by the depth of the query pixel.

Installation
------------

### Dependencies ###

To build the C++ library and the binaries, you will need:

  - cmake (and cmake-curses-gui for easy configuration)
  - [ndarray][ndarray]
  - GCC 4.6 or higher
  - Boost 1.46 or higher
  - NVIDIA CUDA™ 5.0 or higher
  - [Thrust][thrust] - included in CUDA since 4.0
  - [Vigra Impex][vigra]
  - Intel TBB
  - [MDBQ][mdbq] (optional, required for [hyperopt][hyperopt] parameter search)


### Building a debug version ###

```bash
mkdir -p build/debug
cd build/debug
cmake -DCMAKE_BUILD_TYPE=Debug ../../
ccmake .             # adjust paths to your system (cuda, thrust, ...)!
make -j
ctest                # run tests to see if it went well
sudo make install
```

### Building a release version ###

```bash
mkdir -p build/release
cd build/release
cmake -DCMAKE_BUILD_TYPE=Release ../../
ccmake .             # adjust paths to your system (cuda, thrust, ...)!
make -j
ctest                # run tests to see if it went well
sudo make install
```

Dataset Format
--------------

Training and prediction requires to load a set of images from a dataset. We
currently only support datasets that contain RGB-D images, as for example
obtained from a Microsoft Kinect stereo camera. RGB-D images have three
channels that encode the color information and one channel for the depth of
each pixel. Depth is the distance of the object to the camera. Note that
stereo cameras such as the Kinect do not guarantee to deliver a valid depth
measure for every pixel in the image. Distance cannot be measured if the
object is occluded for one of the two cameras. Missing or invalid distance is
either encoded with a zero value or by using the special floating point value `NaN`.

To load images from disk, we use a similar format as the [RGB-D object dataset
of Kevin Lai et al.][lai-rgbd].

We expect to find the color image, depth information and the ground truth in three files in the same folder.
All images must have the same size. Datasets with varying image sizes must be padded manually.
You can specify to skip the padding color when sampling the dataset by using the `--ignoreColor` parameter.

The filename schema and format is

- `<name>_colors.png`
	A three-channel `uint8` RGB image where pixels take on values between 0-255
- `<name>_depth.png`
	A single-channel `uint16` depth image. Each pixel gives
	the depth in millimeters, with 0 denoting missing depth. The depth image can be
	read using MATLAB with the standard function ([imread][matlab-imread]), and in OpenCV by loading
	it into an image of type `IPL_DEPTH_16U`.
- `<name>_ground_truth.png`
	A three-channel `uint8` RGB image where pixels take on values between 0-255.
	Each color represents a different class label. Black indicates "void" or
	"background".

Usage
-----

TODO: documentation

### Training ###

#### Parameters ####

### Prediction ###

#### Parameters ####

### As C++ Library ###

### Parameter Search with [Hyperopt][hyperopt] ###

#### Example ####


How To Implement New Features
-----------------------------

TODO: documentation


[lai-rgbd]: http://www.cs.washington.edu/rgbd-dataset/trd5326jglrepxk649ed/rgbd-dataset_full/README.txt
[ndarray]: https://github.com/deeplearningais/ndarray
[thrust]: http://code.google.com/p/thrust/
[mdbq]: https://github.com/temporaer/MDBQ
[hyperopt]: https://github.com/jaberg/hyperopt
[vigra]: http://hci.iwr.uni-heidelberg.de/vigra/
[matlab-imread]: http://www.mathworks.de/de/help/matlab/ref/imread.html
