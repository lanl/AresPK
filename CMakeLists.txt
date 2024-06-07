# ares - a finite volume, compressible hydrodynamics miniapp using Parthenon
# Copyright (c) 2020-2023, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

cmake_minimum_required(VERSION 3.16)

# Imports machine-specific configuration
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
  include(external/parthenon/cmake/MachineCfg.cmake)
else()
  message(WARNING
    "Could not find MachineCfg.cmake to process machine config file."
    "If you're using Parthenon as submodule and see this message, please report this bug on GiHub."
    "Using a prebuilt Parthenon library with Parthenon-Hydro is currently not tested."
    "For further directions open an issue on GitHub.")
endif()

project(ParthenonHydro LANGUAGES CXX)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

# make paths in target_source() absolute
# used for sources in src/pgen folder
# new behavior introduces in CMake 3.13 that's why this is the min. version above
cmake_policy(SET CMP0076 NEW)

# Don't allow in-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR
      "You cannot build in a source directory (or any directory with a CMakeLists.txt file). "
      "Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

# If the user doesn't specify a build type, prefer Release
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include(cmake/Format.cmake)

# Copy files from source directory to destination directory, substituting any
# variables.  Create destination directory if it does not exist.
macro(configure_files srcDir destDir)
    message(STATUS "Configuring directory ${destDir}")
    make_directory(${destDir})

    file(GLOB templateFiles RELATIVE ${srcDir} "${srcDir}/*")
    foreach(templateFile ${templateFiles})
        set(srcTemplatePath ${srcDir}/${templateFile})
        if(NOT IS_DIRECTORY ${srcTemplatePath})
            message(STATUS "Configuring file ${templateFile}")
            configure_file(
                    ${srcTemplatePath}
                    ${destDir}/${templateFile}
                    COPYONLY)
        endif(NOT IS_DIRECTORY ${srcTemplatePath})
    endforeach(templateFile)
endmacro(configure_files)

# Configure data directory
configure_files(${CMAKE_CURRENT_SOURCE_DIR}/data data)

set(Kokkos_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos)

option(Ares_ENABLE_TESTING "Enable Ares test" ON)
set(PARTHENON_ENABLE_PYTHON_MODULE_CHECK ${Ares_ENABLE_TESTING} CACHE BOOL "Check if local python version contains all modules required for running tests.")

set(PARTHENON_ENABLE_TESTING OFF CACHE BOOL "Disable Parthenon testing.")
set(PARTHENON_DISABLE_OPENMP ON CACHE BOOL "Disable OpenMP")
set(PARTHENON_DISABLE_EXAMPLES ON CACHE BOOL "Don't build Parthenon examples.")
set(PARTHENON_DISABLE_SPARSE ON CACHE BOOL "Disable sparse (not used here)")

# Temporarily also look for MPI here. Ideally, this would be handles from Parthenon
# but the variables are not propagated upwards, e.g., the MPI_FOUND is unknown but
# required for setting up parallel tests.
set(ENABLE_MPI OFF)
set(NUM_MPI_PROC_TESTING "4" CACHE STRING "Number of mpi processors to use when running tests with MPI")
if (NOT PARTHENON_DISABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if (NOT MPI_FOUND)
    message(FATAL_ERROR "MPI is required but couldn't be found. "
    "If you want to build Parthenon without MPI, please rerun CMake with -DPARTHENON_DISABLE_MPI=ON")
  endif()
  set(ENABLE_MPI ON)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/kokkos kokkos)
else()
  find_package(kokkos REQUIRED)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/parthenon parthenon)
else()
  find_package(parthenon REQUIRED)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/singularity-eos singularity-eos)
else()
  find_package(singularity-eos REQUIRED)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/plog/CMakeLists.txt)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/plog plog)
else()
  find_package(plog REQUIRED)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src)

if (Ares_ENABLE_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(tst/regression)
endif()

