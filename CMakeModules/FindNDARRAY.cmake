# This module finds an installed NDARRAY package.
#
# It sets the following variables:
#  NDARRAY_FOUND          - Set to false, or undefined, if NDARRAY isn't found.
#  NDARRAY_INCLUDE_DIRS   - NDARRAY include directories
#  NDARRAY_LIBRARIES      - NDARRAY libraries

FIND_PATH(NDARRAY_INCLUDE_DIR cuv/ndarray.hpp NO_DEFAULT_PATH PATHS ${NDARRAY_ROOT}/src /usr/local/include /usr/include)
FIND_LIBRARY(NDARRAY_LIBRARY ndarray NO_DEFAULT_PATH PATHS ${NDARRAY_ROOT}/build/src/cuv ${NDARRAY_ROOT}/build/release/src/cuv /usr/local/lib /usr/lib)

set(NDARRAY_LIBRARIES ${NDARRAY_LIBRARY} )
set(NDARRAY_INCLUDE_DIRS ${NDARRAY_INCLUDE_DIR})

# handle the QUIETLY and REQUIRED arguments and set NDARRAY_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(NDARRAY DEFAULT_MSG NDARRAY_LIBRARY NDARRAY_INCLUDE_DIR)
IF(NDARRAY_FOUND)
	IF (NOT NDARRAY_FIND_QUIETLY)
		MESSAGE(STATUS "  NDARRAY include dirs: ${NDARRAY_INCLUDE_DIRS}")
		MESSAGE(STATUS "  NDARRAY libraries:    ${NDARRAY_LIBRARIES}")
    ENDIF()
ENDIF()


MARK_AS_ADVANCED( NDARRAY_INCLUDE_DIR NDARRAY_LIBRARY NDARRAY_LIBRARY_DIR)
