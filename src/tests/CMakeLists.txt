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
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/curfil)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src)

SET (TEST_LINK_LIBS curfil)

ADD_EXECUTABLE(image_test image_test.cpp)
TARGET_LINK_LIBRARIES(image_test ${TEST_LINK_LIBS})

CUDA_ADD_EXECUTABLE(random_tree_test random_tree_test.cpp)
TARGET_LINK_LIBRARIES(random_tree_test ${TEST_LINK_LIBS})

CUDA_ADD_EXECUTABLE(random_tree_image_gpu_test random_tree_image_gpu_test.cu)
TARGET_LINK_LIBRARIES(random_tree_image_gpu_test ${TEST_LINK_LIBS})

CUDA_ADD_EXECUTABLE(image_cache_test image_cache_test.cu)
TARGET_LINK_LIBRARIES(image_cache_test ${TEST_LINK_LIBS})

ADD_EXECUTABLE(feature_generation_test feature_generation_test.cpp)
TARGET_LINK_LIBRARIES(feature_generation_test ${TEST_LINK_LIBS})

ADD_EXECUTABLE(import_export_test import_export_test.cpp)
TARGET_LINK_LIBRARIES(import_export_test ${TEST_LINK_LIBS})

ADD_EXECUTABLE(random_tree_image_test random_tree_image_test.cpp)
TARGET_LINK_LIBRARIES(random_tree_image_test ${TEST_LINK_LIBS})

ADD_TEST(image_test "${CMAKE_BINARY_DIR}/src/tests/image_test")
ADD_TEST(feature_generation_test "${CMAKE_BINARY_DIR}/src/tests/feature_generation_test")
ADD_TEST(random_tree_test "${CMAKE_BINARY_DIR}/src/tests/random_tree_test")
ADD_TEST(random_tree_image_gpu_test "${CMAKE_BINARY_DIR}/src/tests/random_tree_image_gpu_test")
ADD_TEST(image_cache_test "${CMAKE_BINARY_DIR}/src/tests/image_cache_test")

ADD_TEST(NAME import_export_test
	COMMAND "${CMAKE_BINARY_DIR}/src/tests/import_export_test" "${CMAKE_SOURCE_DIR}/src/testdata")

ADD_TEST(NAME random_tree_image_test
	COMMAND "${CMAKE_BINARY_DIR}/src/tests/random_tree_image_test" "${CMAKE_SOURCE_DIR}/src/testdata")

IF(MDBQ_FOUND)
	ADD_EXECUTABLE(hyperopt_test hyperopt_test.cpp)
	TARGET_LINK_LIBRARIES(hyperopt_test ${TEST_LINK_LIBS})

	ADD_TEST(hyperopt_test "${CMAKE_BINARY_DIR}/src/tests/hyperopt_test")
ENDIF()
