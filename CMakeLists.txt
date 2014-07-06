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
cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

#
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take their
# definition # and dump it in the cache along with proper documentation,
# otherwise set CMAKE_BUILD_TYPE # to Debug prior to calling PROJECT()
#
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

PROJECT(curfil CXX C)
SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/CMakeModules )

ENABLE_TESTING()
add_subdirectory(src)

FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
	CONFIGURE_FILE(${CMAKE_SOURCE_DIR}/doc/Doxyfile.in ${CMAKE_BINARY_DIR}/docs/Doxyfile)
	ADD_CUSTOM_COMMAND(
		DEPENDS ${CMAKE_BINARY_DIR}/docs/Doxyfile
		OUTPUT  ${CMAKE_BINARY_DIR}/docs/html/index.html
		COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/docs/Doxyfile
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
		)
    ADD_CUSTOM_TARGET(doc DEPENDS ${CMAKE_BINARY_DIR}/docs/html/index.html ${CMAKE_BINARY_DIR}/docs/Doxyfile)
ENDIF(DOXYGEN_FOUND)

include(GetGitRevisionDescription)
git_describe(VERSION --tags --dirty --long)
message(STATUS "GIT VERSION: " ${VERSION})
string(REGEX REPLACE "^v([0-9]+)\\..*" "\\1" VERSION_MAJOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.([0-9]+).*" "\\1" VERSION_MINOR "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+-([0-9]+)-.*" "\\1" VERSION_PATCH "${VERSION}")
string(REGEX REPLACE "^v[0-9]+\\.[0-9]+-[0-9]+-([a-z0-9]+).*" "\\1" VERSION_SHA "${VERSION}")
string(REGEX MATCH "-dirty" VERSION_DIRTY "${VERSION}")
set(VERSION_LONG "${VERSION_MAJOR}.${VERSION_MINOR}-${VERSION_PATCH}-${VERSION_SHA}${VERSION_DIRTY}")
message(STATUS "GIT VERSION: `${VERSION_LONG}'")


ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_BINARY_DIR}/curfil-${VERSION_LONG}-doc.tar.gz
    COMMAND ln -sTf doc curfil-${VERSION_LONG}-doc
    COMMAND tar chzvf curfil-${VERSION_LONG}-doc.tar.gz curfil-${VERSION_LONG}-doc
    DEPENDS ${CMAKE_BINARY_DIR}/docs/html/index.html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_BINARY_DIR}/curfil-${VERSION_LONG}.tar.gz
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
    COMMAND ln -sTf ${CMAKE_SOURCE_DIR} curfil-${VERSION_LONG}
    COMMAND tar chzvf ${CMAKE_BINARY_DIR}/curfil-${VERSION_LONG}.tar.gz --exclude-vcs --exclude 'curfil-${VERSION_LONG}/build*' curfil-${VERSION_LONG}
    )
ADD_CUSTOM_TARGET(dist 
    DEPENDS 
    ${CMAKE_BINARY_DIR}/curfil-${VERSION_LONG}-doc.tar.gz
    ${CMAKE_BINARY_DIR}/curfil-${VERSION_LONG}.tar.gz
    )
