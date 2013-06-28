Accelerated Random Forests for Object-class Image Segmentation
--------------------------------------------------------------

This project is an open source implementation with NVIDIA CUDA™ that accelerates random
forest training and prediction for image segmentation applications by using the
massive parallel computing power offered by GPUs.


Installation
------------

### Dependencies ###

To build the C++ lib, you will need:

  - cmake (and cmake-curses-gui for easy configuration)
	- [ndarray][ndarray]
  - libboost-dev >= 1.37
  - NVIDIA CUDA (tm), including SDK. We support versions 3.X, 4.X and 5.X
	- GCC 4.6 or higher
	- NVIDIA CUDA™ 5.0 or higher
	- [Vigra Impex][vigra]
	- Intel TBB
	-	[Thrust][thrust]
	- Boost 1.46 or higher
	- [MDBQ][mdbq] (optional, required for [hyperopt][hyperopt] parameter search)
  - [thrust library][thrust] - included in CUDA since 4.0


### Building a debug version ###

```bash
mkdir -p build/debug
cd build/debug
cmake -DCMAKE_BUILD_TYPE=Debug ../../
make
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


Implemented Visual Features
---------------------------

This project focuses on image segmentation and classification applications.

We implement two types of RGB-D image features.

For a given query pixel, the image feature is calculated as the difference of
the average value of the image channel in two rectangular regions in the
neighborhood around the query pixel.

Extent size and relative offset of the rectangular region in the image is
normalized by the depth of the query pixel.

This leads to the property that smaller regions and offsets are used for
pixels that have a larger distance (e.g. pixels in the background).

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
	The filename schema and format is the following:

### Color Image ###
  - `<name>_colors.png`
	A three-channel `uint8` RGB image where pixels take on values between 0-255

### Depth Image ###
	- `<name>_depth.png`
	A single-channel `uint16` depth image. Each pixel gives
	the depth in millimeters, with 0 denoting missing depth. The depth image can be
	read using MATLAB with the standard function (imread), and in OpenCV by loading
	it into an image of type `IPL_DEPTH_16U`.

### Ground Truth ###
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
