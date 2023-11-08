# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    ares
        main.cpp
        ares_driver.cpp
        hydro/hydro.cpp
        network/network.cpp
        recon/plm_simple.hpp
        recon/ppm_simple.hpp
        utils/utils.hpp
        refinement/gradient.cpp
        refinement/other.cpp
        utils/utils.cpp
)

add_subdirectory(pgen)

target_link_libraries(ares PRIVATE plog)
target_link_libraries(ares PRIVATE parthenon)
target_link_libraries(ares PRIVATE singularity-eos)
