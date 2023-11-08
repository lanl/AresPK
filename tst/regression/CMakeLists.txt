# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

list(APPEND REQUIRED_PYTHON_MODULES numpy unyt matplotlib h5py scipy)
message(STATUS
  "Checking for Python modules (${REQUIRED_PYTHON_MODULES}) required for regression tests.\n"
  "   Check can be disabled by setting PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF but then\n"
  "   tests are not guaranteed to work anymore.")

# import Parthenon setup_test_serial and setup_test_parallel
include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)


function(setup_test_both dir arg test_labels)
  setup_test_serial(${dir} "${arg}" "${test_labels}")
  # just using 2 ranks for MPI tests
  setup_test_parallel(2 ${dir} "${arg}" "${test_labels}")
endfunction()

setup_test_both("riemann_hydro" "--driver ${PROJECT_BINARY_DIR}/bin/ares \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sod.in --num_steps 20" "other")

setup_test_both("convergence" "--driver ${PROJECT_BINARY_DIR}/bin/ares \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 12" "convergence")
