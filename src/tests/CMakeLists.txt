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
	COMMAND "${CMAKE_BINARY_DIR}/src/tests/import_export_test"
	WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/src")

ADD_TEST(NAME random_tree_image_test
	COMMAND "${CMAKE_BINARY_DIR}/src/tests/random_tree_image_test"
	WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/src")

IF(MDBQ_FOUND)
	ADD_EXECUTABLE(hyperopt_test hyperopt_test.cpp)
	TARGET_LINK_LIBRARIES(hyperopt_test ${TEST_LINK_LIBS})

	ADD_TEST(hyperopt_test "${CMAKE_BINARY_DIR}/src/tests/hyperopt_test")
ENDIF()
