cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(cornValidation)

#set(Boost_USE_STATIC_LIBS ON)
#set(Boost_USE_MULTITHREADED ON)
#ADD_DEFINITIONS(-DBoost_USE_STATIC_LIBS=ON)
#set(Boost_INCLUDE_DIR /usr/include/)
#set(Boost_LIBRARY_DIR /usr/lib/x86_64-linux-gnu/)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")


#find_package(Boost REQUIRED)

find_package(PCL)
find_package(OpenCV)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#add_executable(cornValidation corn2nd/corn.cpp)
#add_executable(cornValidation maize04132019/corn.cpp)
#add_executable(cornValidation corn2nd/paperVisual.cpp)
add_executable(cornValidation babcorn/corn.cpp)
#add_executable(cornValidation babcorn/corn-6-25.cpp)
target_link_libraries(cornValidation ${PCL_LIBRARIES} ${OpenCV_LIBS})
