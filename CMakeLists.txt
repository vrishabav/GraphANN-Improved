cmake_minimum_required(VERSION 3.14)
project(graphann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags (compiler-specific)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# ---- Vamana library (shared sources) ----
add_library(graphann_lib STATIC
    src/distance.cpp
    src/io_utils.cpp
    src/vamana_index.cpp
)
target_include_directories(graphann_lib PUBLIC include)
target_link_libraries(graphann_lib PUBLIC OpenMP::OpenMP_CXX)

# ---- HNSW library ----
add_library(hnsw_lib STATIC
    src/hnsw_index.cpp
)
target_include_directories(hnsw_lib PUBLIC include)
# HNSW uses distance.cpp and io_utils.cpp from graphann_lib
target_link_libraries(hnsw_lib PUBLIC graphann_lib)

# ---- Vamana executables ----

# Build Vamana index
add_executable(build_index src/build_index.cpp)
target_link_libraries(build_index PRIVATE graphann_lib)

# Search Vamana index
add_executable(search_index src/search_index.cpp)
target_link_libraries(search_index PRIVATE graphann_lib)

# Hard query analysis tool
add_executable(hard_query_analysis src/hard_query_analysis.cpp)
target_link_libraries(hard_query_analysis PRIVATE graphann_lib)

# ---- HNSW executables ----

# Build HNSW index
add_executable(build_hnsw src/build_hnsw.cpp)
target_link_libraries(build_hnsw PRIVATE hnsw_lib)

# Search HNSW index
add_executable(search_hnsw src/search_hnsw.cpp)
target_link_libraries(search_hnsw PRIVATE hnsw_lib)
