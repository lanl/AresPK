// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <libgen.h>
#include <limits> // numeric limits
#include <linux/limits.h>
#include <stdio.h>
#include <string.h>

#include "basic_types.hpp" // Real

// TODO(pgrete) There's a compiler bug in nvcc < 11.2 that precludes the use
// of C++17 with relaxed-constexpr in Kokkos,
// see https://github.com/kokkos/kokkos/issues/3496
// This also precludes our downstream use of constexpr int here.
// Update once nvcc/cuda >= 11.2 is more widely available on machine.
enum {
  IDN = 0,
  IM1 = 1,
  IM2 = 2,
  IM3 = 3,
  IEN = 4,
  NHYDRO = 5,
  NXNUC = 55,
};

// array indices for 1D primitives: velocity and pressure
enum { IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4 };

enum class RiemannSolver { undefined, none, hlle, hllc };
enum class Fluid { undefined, euler };
enum class Reconstruction { undefined, plm, ppm };

// nse -> Nuclear Statistical Equilibrium Solver
enum class NetworkSolver { undefined, none, nse };

enum class GravitySolver { undefined, none, monopole, constant };

enum class Hst { idx, ekin };

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

#endif // MAIN_HPP_
