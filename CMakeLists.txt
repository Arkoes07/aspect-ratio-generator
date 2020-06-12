cmake_minimum_required (VERSION 3.8)

project ("AspectRatioGenerator")

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

add_executable (AspectRatioGenerator "AspectRatioGenerator.cpp"  "LandmarkPredictor.cpp"  "LandmarkPredictor.h")

target_link_libraries (AspectRatioGenerator ${OpenCV_LIBS} dlib::dlib)
