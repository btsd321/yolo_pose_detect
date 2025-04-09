cmake_minimum_required(VERSION 3.18)
project(yolo_pose_detect)

string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")

# #onnxruntime providers
# option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
# option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
# option(ONNXRUNTIME_ROOTDIR "~/Library/onnxruntime/onnxruntime-linux-x64-1.21.0")
# include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# if(NOT ONNXRUNTIME_ROOTDIR)
#   if(WIN32)
#     set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
# else()
#     set(ONNXRUNTIME_ROOTDIR "~/Library/onnxruntime/onnxruntime-linux-x64-1.21.0")
#   endif()
# endif()

# if(onnxruntime_USE_CUDA)
#   add_definitions(-DUSE_CUDA)
# endif()
# if(onnxruntime_USE_TENSORRT)
#   add_definitions(-DUSE_TENSORRT)
# endif()
# if(onnxruntime_USE_DML)
#   message("Enabling DML")
#   add_definitions(-DUSE_DML)
# endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# find_package(onnxruntime REQUIRED CONFIG PATHS ${ONNXRUNTIME_ROOTDIR}/lib64/cmake/onnxruntime)
find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX PATHS /usr/lib/cmake/openvino2025.0.0)

include_directories(
    # "${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
    # "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
    # "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session"  # Windows local install
    # opencv
    ${OpenCV_INCLUDE_DIRS} 
    # eigen
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link_directories("${ONNXRUNTIME_ROOTDIR}/lib64")

file(GLOB SRC_FILES "src/*.cpp")
file(GLOB INCLUDE_FILES "include/*.h")

set(PROJECT_FILES ${SRC_FILES} ${INCLUDE_FILES})

add_executable(${PROJECT_NAME} ${PROJECT_FILES})

target_link_libraries(${PROJECT_NAME} 
    # onnxruntime
    openvino::frontend::onnx 
    openvino::runtime
    ${EIGEN3_LIBS}
    ${OpenCV_LIBRARIES})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/data $<TARGET_FILE_DIR:${PROJECT_NAME}>/data)