# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
if (NOT APPLE)
  find_package(OpenMP)
endif ()

find_package(Doxygen)

# ============================================================================ #
# Compilation flags
IF(UNIX)
  SET(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -g -O0  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
  SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g  -Wall -Wextra -Wunused-variable -DDEBUG -D_DEBUG")
ENDIF(UNIX)

if(OPENMP_FOUND)
  MESSAGE("OpenMP found")
  if(UNIX)
    SET(CMAKE_C_FLAGS_RELEASE "-O3  -Wall -Wextra -Wunused-variable  -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -Wextra -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
  endif(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  else(OPENMP_FOUND)
    MESSAGE("OpenMP not found")
    if(UNIX)
      SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
      SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wno-unknown-pragmas -Wunused-variable -g -fPIC -msse2 -msse3 -msse4 -ffast-math")
    endif(UNIX)
endif(OPENMP_FOUND)

# ============================================================================ #
# Create projects
include_directories("lib/")

# KAZE library
set(KAZE_SRCS
    lib/KAZEConfig.h
    lib/KAZE.h                   lib/KAZE.cpp
    lib/fed.h                    lib/fed.cpp
    lib/nldiffusion_functions.h  lib/nldiffusion_functions.cpp
    lib/utils.h                  lib/utils.cpp)

add_library(KAZE ${KAZE_SRCS})
target_link_libraries(KAZE ${OpenCV_LIBS})

# Feature detector program
add_executable(kaze_features kaze_features.cpp)
target_link_libraries(kaze_features KAZE)

# Image matching program
add_executable(kaze_match kaze_match.cpp)
target_link_libraries(kaze_match KAZE)

# Program that compares KAZE with SIFT and SURF
add_executable(kaze_compare kaze_compare.cpp)
target_link_libraries(kaze_compare KAZE)

# ============================================================================ #
# Library installation
install(TARGETS KAZE DESTINATION ${KAZE_INSTALL_PREFIX})
install(FILES
    lib/KAZE.h
    lib/fed.h
    lib/utils.h
    lib/nldiffusion_functions.h
    lib/KAZEConfig.h
    DESTINATION
    ${KAZE_INCLUDE_PREFIX}
)

# ============================================================================ #
# Generate Doxygen-based documentation project
if(DOXYGEN_FOUND)
  add_custom_target(akaze_documentation
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)
