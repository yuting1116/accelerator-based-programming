# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/yuting/assignment3/builddir/_deps/kokkos-src")
  file(MAKE_DIRECTORY "/home/yuting/assignment3/builddir/_deps/kokkos-src")
endif()
file(MAKE_DIRECTORY
  "/home/yuting/assignment3/builddir/_deps/kokkos-build"
  "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix"
  "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/tmp"
  "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp"
  "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/src"
  "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/gorilla/home/yuting/assignment3/builddir/_deps/kokkos-subbuild/kokkos-populate-prefix/src/kokkos-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
