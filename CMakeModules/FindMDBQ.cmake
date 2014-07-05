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
# This module finds an installed MDBQ package.
#
# It sets the following variables:
#  MDBQ_FOUND          - Set to false, or undefined, if MDBQ isn't found.
#  MDBQ_INCLUDE_DIRS   - MDBQ include directories
#  MDBQ_LIBRARIES      - MDBQ libraries

FIND_PATH(MDBQ_INCLUDE_DIR mdbq/client.hpp PATHS ${MDBQ_ROOT}/src /usr/local/include /usr/include)
FIND_LIBRARY(MDBQ_LIBRARY mdbq PATHS ${MDBQ_ROOT}/build/src/mdbq /usr/local/lib /usr/lib)

FIND_PATH(BSON_INCLUDE_DIR mongo/bson/bson.h)

set(MDBQ_LIBRARIES ${MDBQ_LIBRARY} )
set(MDBQ_INCLUDE_DIRS ${MDBQ_INCLUDE_DIR} ${BSON_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set MDBQ_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MDBQ DEFAULT_MSG MDBQ_LIBRARY MDBQ_INCLUDE_DIR)
IF(MDBQ_FOUND)
    IF (NOT MDBQ_FIND_QUIETLY)
        MESSAGE(STATUS "  MDBQ include dirs: ${MDBQ_INCLUDE_DIRS}")
        MESSAGE(STATUS "  MDBQ libraries:    ${MDBQ_LIBRARIES}")
    ENDIF()
ENDIF()


MARK_AS_ADVANCED( MDBQ_INCLUDE_DIR MDBQ_LIBRARY MDBQ_LIBRARY_DIR)
