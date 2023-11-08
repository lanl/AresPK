//! \file rt.cpp
//! \brief Problem generator for RT instability.

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>

// AthenaPK headers
#include "../main.hpp"

namespace rt {
using namespace parthenon::driver::prelude;

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Rayleigh-Taylor test

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto gam = pin->GetReal("hydro", "gamma");
  auto mode = pin->GetOrAddString("problem/rt", "pert_mode", "none");
  auto gm1 = (gam - 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &v_dev = rc->Get("prim").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  auto v = v_dev.GetHostMirrorAndCopy();

  std::mt19937 gen(pmb->gid); // Standard mersenne_twister_engine seeded with gid
  std::uniform_real_distribution<Real> ran(-1.0, 1.0);

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        if (coords.Xc<1>(i) <= 0.0) {
          u(IDN, k, j, i) = 1.0;
        } else {
          u(IDN, k, j, i) = 3.0;
        }

        if (mode == "single") {
          u(IM1, k, j, i) = u(IDN, k, j, i) * 0.1 *
                            (1 + cos(3 * M_PI * coords.Xc<1>(i))) *
                            (1 + cos(4 * M_PI * coords.Xc<2>(j))) *
                            (1 + cos(4 * M_PI * coords.Xc<3>(j))) / 4.0;
        } else if (mode == "multiple") {
          u(IM1, k, j, i) = u(IDN, k, j, i) * 0.05 * ran(gen) *
                            (1 + cos(2 * M_PI * coords.Xc<1>(i)) / 1.5);
        } else if (mode == "none") {
          u(IM1, k, j, i) = 0.0;
        } else if (mode == "constant") {
          u(IM1, k, j, i) = 0.1;
        }
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        v(IPR, k, j, i) = 0.6 - 0.1 * u(IDN, k, j, i) * coords.Xc<1>(i);

        u(IEN, k, j, i) =
            v(IPR, k, j, i) / gm1 +
            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);

      } // for i
    }   // for j
  }     // for k

  u_dev.DeepCopy(u);
  v_dev.DeepCopy(v);

} // ProblemGenerator
} // namespace rt
