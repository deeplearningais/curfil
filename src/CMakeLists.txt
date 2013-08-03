cmake_minimum_required( VERSION 2.6 FATAL_ERROR )

FIND_PACKAGE(CUDA     REQUIRED)
FIND_PACKAGE(TBB      REQUIRED)
FIND_PACKAGE(VIGRA    REQUIRED)
FIND_PACKAGE(MDBQ)

SET(CMAKE_CXX_COMPILER g++-4.6)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wformat=2")

SET(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -O0 -G")

ADD_DEFINITIONS("-std=c++0x")

# http://stackoverflow.com/questions/8450030/can-modern-c-compilers-inline-functions-that-are-defined-in-a-cpp-file
ADD_DEFINITIONS("-flto")

# ---------- Find Boost Headers/Libraries -----------------------
SET (Boost_FIND_REQUIRED TRUE)
SET (Boost_FIND_QUIETLY FALSE)
SET (Boost_USE_MULTITHREADED TRUE)
SET (Boost_USE_STATIC_LIBS FALSE)
FIND_PACKAGE(Boost 1.46 COMPONENTS system filesystem iostreams program_options date_time REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

FIND_PATH(THRUST_PATH thrust/device_vector.h /usr/include /usr/local/include ${CUDA_INCLUDE_DIRS} "$ENV{THRUST_ROOT}")
IF(NOT THRUST_PATH)
	MESSAGE(FATAL_ERROR "Could not find the thrust library. Please install in standard locations or set THRUST_ROOT environment variable.")
ENDIF(NOT THRUST_PATH)

if ( NOT CUDA_ARCHITECTURE )
	SET( CUDA_ARCHITECTURE -gencode;arch=compute_20,code=sm_20;-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_35,code=sm_35 )
endif()

SET(CUDA_ARCHITECTURE ${CUDA_ARCHITECTURE} CACHE STRING "The CUDA architecture to compile for, i.e. -gencode;arch=compute_2,code=sm_20")
SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCHITECTURE}")
MESSAGE(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

CUDA_INCLUDE_DIRECTORIES( ${THRUST_PATH}       )
INCLUDE_DIRECTORIES(      ${THRUST_PATH}       )

CUDA_INCLUDE_DIRECTORIES( ${TBB_INCLUDE_DIRS}  )
INCLUDE_DIRECTORIES(      ${TBB_INCLUDE_DIRS}  )

CUDA_INCLUDE_DIRECTORIES( ${VIGRA_INCLUDE_DIR} )
INCLUDE_DIRECTORIES(      ${VIGRA_INCLUDE_DIR} )

CUDA_INCLUDE_DIRECTORIES( ${THRUST_PATH}                                )
INCLUDE_DIRECTORIES(      ${THRUST_PATH}                                )

CUDA_INCLUDE_DIRECTORIES( ${CMAKE_SOURCE_DIR}/src/ndarray/src )
INCLUDE_DIRECTORIES(      ${CMAKE_SOURCE_DIR}/src/ndarray/src )

IF(MDBQ_FOUND)
	CUDA_INCLUDE_DIRECTORIES( ${MDBQ_INCLUDE_DIRS} )
	INCLUDE_DIRECTORIES(      ${MDBQ_INCLUDE_DIRS} )
ELSE()
SET (MDBQ_FILES )
ENDIF()

add_subdirectory(curfil)
add_subdirectory(ndarray/src/cuv)
add_subdirectory(tests)

# INSTALL(DIRECTORY curfil${LIB_SUFFIX}/
# 	DESTINATION "include/curfil"
# 	FILES_MATCHING PATTERN "*.hpp"
# )
# IF("${LIB_SUFFIX}" STREQUAL "")
#     INSTALL(FILES ndarray.hpp DESTINATION include)
# ENDIF("${LIB_SUFFIX}" STREQUAL "")

CUDA_BUILD_CLEAN_TARGET()
