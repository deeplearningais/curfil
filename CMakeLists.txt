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
