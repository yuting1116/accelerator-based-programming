cmake_minimum_required(VERSION 3.22)
project(assignment3)

include(FetchContent)
FetchContent_Declare(
  Kokkos
  URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
)
FetchContent_MakeAvailable(Kokkos)
add_executable(HelloKokkos jacobian.cpp)
target_link_libraries(HelloKokkos Kokkos::kokkos)