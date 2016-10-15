cmake_minimum_required (VERSION 3.0)
project (eigenVsAither)

add_executable (eigenVsAither main.cpp)

set_property (TARGET eigenVsAither PROPERTY CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

# set additional c++ flags for all build types
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -march=native -O3 -DNDEBUG")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -pedantic -march=native")
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -pedantic -O0 -ggdb -pg")

# set compiler flags based on release type
if (CMAKE_BUILD_TYPE MATCHES release)
   message (STATUS "C++ flags: " ${CMAKE_CXX_FLAGS_RELEASE})
elseif (CMAKE_BUILD_TYPE MATCHES debug)
   message (STATUS "C++ flags: " ${CMAKE_CXX_FLAGS_DEBUG})
else ()
   message (STATUS "C++ flags: " ${CMAKE_CXX_FLAGS})
endif ()



# get aither includes and libraries
if (AITHER_DIR)
  message (STATUS "Looking for aither includes and libraries here: ${AITHER_DIR}")
else ()
  message (STATUS "AITHER_DIR not specified, looking for aither in /usr/local")
  set (AITHER_DIR /usr/local)
endif ()

include_directories (${AITHER_DIR}/include)
target_link_libraries (eigenVsAither ${AITHER_DIR}/lib/libaither.so)

# get eigen
if (EIGEN_DIR)
  message (STATUS "Looking for eigen includes here: ${EIGEN_DIR}")
else ()
  message (STATUS "EIGEN_DIR not specified, looking for aither in /usr/local/include/eigen3")
  set (EIGEN_DIR /usr/local/include/eigen3)
endif ()

include_directories (/usr/local/include/eigen3)

install (TARGETS eigenVsAither DESTINATION bin)
