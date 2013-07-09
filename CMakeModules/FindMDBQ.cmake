# This module finds an installed MDBQ package.
#
# It sets the following variables:
#  MDBQ_FOUND          - Set to false, or undefined, if MDBQ isn't found.
#  MDBQ_INCLUDE_DIRS   - MDBQ include directories
#  MDBQ_LIBRARIES      - MDBQ libraries

FIND_PATH(MDBQ_INCLUDE_DIR mdbq/client.hpp NO_DEFAULT_PATH PATHS ${MDBQ_ROOT}/src /usr/local/include /usr/include)
FIND_LIBRARY(MDBQ_LIBRARY mdbq NO_DEFAULT_PATH PATHS ${MDBQ_ROOT}/build/src/mdbq /usr/local/lib /usr/lib)

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
