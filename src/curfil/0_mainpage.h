#if 0
#######################################################################################
# The MIT License

# Copyright (c) 2014       Hannes Schulz, University of Bonn  <schulz@ais.uni-bonn.de>
# Copyright (c) 2013       Benedikt Waldvogel, University of Bonn <mail@bwaldvogel.de>
# Copyright (c) 2008-2009  Sebastian Nowozin                       <nowozin@gmail.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#######################################################################################
#endif

/**
 *
 *
 * @defgroup forest_hierarchy Forest hierarchy
 * @defgroup hyperopt Hyperopt
 * @defgroup import_export_trees Import and Export trees
 * @defgroup image_related Image related classes
 * @defgroup LRU_cache LRU Cache
 * @defgroup prediction_results Prediction Results
 * @defgroup rand_sampling Random Sampling
 * @defgroup score_calc Score Calculation
 * @defgroup transfer_cpu_gpu Transfer between CPU and GPU
 * @defgroup training_helper_classes Training helper classes
 * @defgroup unit_testing Unit Testing
 *
 *
 * @mainpage
 * CUDA Random Forests for Image Labeling (CURFIL)
 * -----------------------------------------------
 *
 * This project is an open source implementation with NVIDIA CUDA that accelerates random
 * forest training and prediction for image labeling by using the
 * massive parallel computing power offered by GPUs.
 *
 * CURFIL is the result of Benedikt Waldvogel’s master thesis
 * [Accelerating Random Forests on CPUs and GPUs for Object-Class Image Segmentation][master-thesis]
 * at the University of Bonn, [Autonomous Intelligent Systems][ais-bonn].
 *
 * Implemented Visual Features
 * ---------------------------
 *
 * We currently focus on image labelling tasks such as image segmentation or classification applications.
 * We implement two types of image features as described in the
 * [documentation of visual features][visual-features] in more detail.
 *
 * Installation
 * ------------
 *
 * ### Dependencies ###
 *
 * To build the C++ library and the binaries, you will need:
 *
 *   - cmake (and cmake-curses-gui for easy configuration)
 *   - [ndarray][ndarray] (included as git submodule)
 *   - GCC 4.4 or higher
 *   - Boost 1.46 or higher
 *   - NVIDIA CUDA™ 5.0 or higher
 *   - [Thrust][thrust] - included in CUDA since 4.0
 *   - [Vigra Impex][vigra]
 *   - Intel TBB
 *   - [MDBQ][mdbq] (optional, required for [hyperopt][hyperopt] parameter search)
 *   - A CUDA capable GPU with compute capability 2.0 or higher
 *
 *
 * ### Building ###
 *
 * @code
 * git clone --recursive https://github.com/deeplearningais/curfil.git  # --recursive will also init the submodules
 * cd curfil
 * mkdir -p build
 * cd build
 * cmake -DCMAKE_BUILD_TYPE=Release ..  # change to 'Debug' to build the debugging version
 * ccmake .              # adjust paths to your system (cuda, thrust, ...)!
 * make -j
 * ctest                 # run tests to see if it went well
 * sudo make install
 * @endcode
 * Refer to your local Unix expert if you do not know what to do with this instruction.
 *
 * Dataset Format
 * --------------
 *
 * Training and prediction requires to load a set of images from a dataset. We
 * currently only support datasets that contain RGB-D images, as for example
 * captured by the Microsoft Kinect or the Asus Xtion PRO LIVE. RGB-D images have
 * three channels that encode the color information and one channel for the depth
 * of each pixel. Depth is the distance of the object to the camera. Note that
 * stereo cameras such as the Kinect do not guarantee to deliver a valid depth
 * measure for every pixel in the image. Distance cannot be measured if the object
 * is occluded for one of the two cameras. Missing or invalid distance is either
 * encoded with a zero value or by using the special floating point value `NaN`.
 *
 * To load images from disk, we use a similar format as the [RGB-D object dataset
 * of Kevin Lai et al.][lai-rgbd].
 *
 * We expect to find the color image, depth information and the ground truth in three files in the same folder.
 * All images must have the same size. Datasets with varying image sizes must be padded manually.
 * You can specify to skip the padding color when sampling the dataset by using the `--ignoreColor` parameter.
 *
 * The filename schema and format is
 *
 * - `name_colors.png`
 * 	A three-channel `uint8` RGB image where pixels take on values between 0-255
 * - `name_depth.png`
 * 	A single-channel `uint16` depth image. Each pixel gives
 * 	the depth in millimeters, with 0 denoting missing depth. The depth image can be
 * 	read using MATLAB with the standard function ([imread][matlab-imread]), and in OpenCV by loading
 * 	it into an image of type `IPL_DEPTH_16U`.
 * - `name_ground_truth.png`
 * 	A three-channel `uint8` RGB image where pixels take on values between 0-255.
 * 	Each color represents a different class label. Black indicates "void" or
 * 	"background".
 *
 * Usage
 * -----
 *
 * ### Training ###
 *
 * Use the binary `curfil_train`.
 *
 * The training process produces a random forest consisting of multiple decision trees
 * that are serialized to compressed JSON files, one file per tree.
 *
 * See the [documentation of training parameters](https://github.com/deeplearningais/curfil/wiki/Training-Parameters).
 *
 * ### Prediction ###
 *
 * Use the binary `curfil_predict`.
 *
 * The program reads the trees from the compresses JSON files and performs a dense
 * pixel-wise classification of the specified input images.
 * Prediction is accelerated on GPU and runs in real-time speed even on mobile
 * GPUs such as the NVIDIA GeForce GTX 675M.
 *
 * Also see [documentation of prediction parameters](http://github.com/deeplearningais/curfil/wiki/Prediction-Parameters).
 *
 * ### Hyperopt Parameter Search ###
 *
 * Use the binary `curfil_hyperopt`.
 *
 * This [Hyperopt][hyperopt] client is only built if [MDBQ][MDBQ] is installed.
 * The client fetches hyperopt trials (jobs) from a MongoDB database and performs 5-fold cross-validation to evaluate the loss.
 * You can run the hyperopt client in parallel on as many machines as desired.
 *
 * The trials need to be inserted into the database in advance.
 * We include sample python scripts in [scripts/](scripts/).
 * Note that there is only one *new* trial in the database at any given point in time.
 * Thus, the python script needs to be running during the entire parameter search.
 *
 * The procedure:
 *
 *  1. Make sure the MongoDB database is up an running.
 *  2. Run the python script that inserts new trials. Example: `scripts/NYU/hyperopt_search.py`
 *  3. Run `curfil_hyperopt` on as many machine as desired.
 *
 * Also see [documentation of hyperopt parameter search in the wiki](http://github.com/deeplearningais/curfil/wiki/Hyperopt-Parameter-Search).
 *
 * ### As a `C++` Library ###
 *
 * See [the example in the wiki](https://github.com/deeplearningais/curfil/wiki/Usage-as-a-Library).
 *
 * Examples
 * --------
 *
 * - [Training and Prediction with the NYU Depth v2 Dataset](https://github.com/deeplearningais/curfil/wiki/Training-and-Prediction-with-the-NYU-Depth-v2-Dataset)
 *
 *
 * [master-thesis]: http://www.ais.uni-bonn.de/theses/Benedikt_Waldvogel_Master_Thesis_07_2013.pdf
 * [ais-bonn]: http://www.ais.uni-bonn.de
 * [visual-features]: https://github.com/deeplearningais/curfil/wiki/Visual-Features
 * [lai-rgbd]: http://www.cs.washington.edu/rgbd-dataset/trd5326jglrepxk649ed/rgbd-dataset_full/README.txt
 * [ndarray]: https://github.com/deeplearningais/ndarray
 * [thrust]: http://code.google.com/p/thrust/
 * [mdbq]: https://github.com/temporaer/MDBQ
 * [hyperopt]: https://github.com/jaberg/hyperopt
 * [vigra]: http://hci.iwr.uni-heidelberg.de/vigra/
 * [matlab-imread]: http://www.mathworks.de/de/help/matlab/ref/imread.html

*/
