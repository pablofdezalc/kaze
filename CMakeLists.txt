# ============================================================================ #
project(KAZE_features)

cmake_minimum_required (VERSION 2.6)
if (COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif ()

# ============================================================================ #
# Build Type. By default is release
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release" FORCE)

# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")

# ============================================================================ #
# Where to install the KAZE library
set(KAZE_INSTALL_PREFIX "/usr/local/kaze/lib" CACHE PATH "Installation Directory")
set(KAZE_INCLUDE_PREFIX "/usr/local/kaze/include" CACHE PATH "Includes Directory")

# ============================================================================ #
# CPP sources
message(STATUS ">>> Adding src subdirectory")
add_subdirectory(src)
