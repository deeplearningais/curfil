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
# This module finds an installed Vigra package.
#
# It sets the following variables:
#  VIGRA_FOUND              - Set to false, or undefined, if vigra isn't found.
#  VIGRA_INCLUDE_DIR        - Vigra include directory.
#  VIGRA_IMPEX_LIBRARY      - Vigra's impex library
#  VIGRA_IMPEX_LIBRARY_DIR  - path to Vigra impex library

# configVersion.hxx only present, after build of Vigra
FIND_PATH(VIGRA_INCLUDE_DIR vigra/configVersion.hxx PATHS $ENV{VIGRA_ROOT}/include ENV CPLUS_INCLUDE_PATH)
FIND_LIBRARY(VIGRA_IMPEX_LIBRARY vigraimpex PATHS $ENV{VIGRA_ROOT}/src/impex $ENV{VIGRA_ROOT}/lib ENV LD_LIBRARY_PATH ENV LIBRARY_PATH)
GET_FILENAME_COMPONENT(VIGRA_IMPEX_LIBRARY_PATH ${VIGRA_IMPEX_LIBRARY} PATH)
SET( VIGRA_IMPEX_LIBRARY_DIR ${VIGRA_IMPEX_LIBRARY_PATH} CACHE PATH "Path to Vigra impex library.")

# handle the QUIETLY and REQUIRED arguments and set VIGRA_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VIGRA DEFAULT_MSG VIGRA_IMPEX_LIBRARY VIGRA_INCLUDE_DIR)
IF(VIGRA_FOUND)
    IF (NOT VIGRA_FIND_QUIETLY)
        MESSAGE(STATUS "  includes:      ${VIGRA_INCLUDE_DIR}")
        MESSAGE(STATUS "  impex library dir: ${VIGRA_IMPEX_LIBRARY_DIR}")
        MESSAGE(STATUS "  impex library: ${VIGRA_IMPEX_LIBRARY}")
    ENDIF()
ENDIF()


MARK_AS_ADVANCED( VIGRA_INCLUDE_DIR VIGRA_IMPEX_LIBRARY VIGRA_IMPEX_LIBRARY_DIR)
