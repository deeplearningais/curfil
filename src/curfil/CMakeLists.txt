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
# http://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake/4318642#4318642
include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cpp.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cpp" @ONLY)

IF(MDBQ_FOUND)
	SET (MDBQ_FILES hyperopt.cpp)
ELSE()
	SET (MDBQ_FILES )
	SET (MDBQ_LIBRARIES )
ENDIF()

CUDA_ADD_LIBRARY(curfil SHARED random_tree_image_gpu.cu random_tree.cpp image.cpp utils.cpp ndarray_ops.cpp random_tree_image.cpp random_forest_image.cpp import.cpp export.cpp predict.cpp ndarray_ops.cpp train.cpp ${MDBQ_FILES} "${CMAKE_CURRENT_BINARY_DIR}/version.cpp")

TARGET_LINK_LIBRARIES(curfil ndarray ${VIGRA_IMPEX_LIBRARY} ${TBB_LIBRARIES} Boost::filesystem Boost::iostreams Boost::date_time ${MDBQ_LIBRARIES} pthread ${CUDA_LIBRARIES} )

INSTALL(TARGETS curfil
	DESTINATION "lib"
)

INSTALL(FILES random_tree.h random_tree_image.h random_forest_image.h image.h score.h random_tree_image_gpu.h predict.h import.h export.h utils.h
	DESTINATION "include/curfil"
)

ADD_EXECUTABLE(curfil_train train_main.cpp)
TARGET_LINK_LIBRARIES(curfil_train curfil Boost::program_options)

ADD_EXECUTABLE(curfil_predict predict_main.cpp)
TARGET_LINK_LIBRARIES(curfil_predict curfil Boost::program_options)

INSTALL(TARGETS curfil_train curfil_predict
    DESTINATION "bin"
)

IF(MDBQ_FOUND)
	ADD_EXECUTABLE(curfil_hyperopt hyperopt_main.cpp)
	TARGET_LINK_LIBRARIES(curfil_hyperopt curfil)
ENDIF()
