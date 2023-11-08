//! \file gas_sphere.cpp
//! \brief Problem generator for a gaseous sphere.

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

namespace gas_sphere {
using namespace parthenon::driver::prelude;

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Rayleigh-Taylor test

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto gam = pin->GetReal("hydro", "gamma");
  auto sp_radius = pin->GetOrAddReal("problem/gas_sphere", "sp_radius", 1.0);
  auto dens_sphere = pin->GetOrAddReal("problem/gas_sphere", "sp_dens", 1.0);
  auto gm1 = (gam - 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &v_dev = rc->Get("prim").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  auto v = v_dev.GetHostMirrorAndCopy();

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real radius2 = SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k));

        if (sqrt(radius2) < sp_radius) {
          u(IDN, k, j, i) = dens_sphere;
        } else {
          u(IDN, k, j, i) = 1.e-3;
        }

        const Real vel_mag = 0.0;
        Real phi = acos(coords.Xc<3>(k) / sqrt(radius2));
        Real cos_phi = cos(phi);
        Real sin_phi = sin(phi);
        Real cos_tht = coords.Xc<1>(i) / sqrt(radius2);
        Real sin_tht = coords.Xc<2>(j) / sqrt(radius2);

        u(IM1, k, j, i) = vel_mag * sin_tht * cos_phi / sqrt(radius2);
        u(IM2, k, j, i) = vel_mag * sin_tht * sin_tht / sqrt(radius2);
        u(IM3, k, j, i) = vel_mag * cos_tht / sqrt(radius2);

        v(IPR, k, j, i) = 1.0;

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
} // namespace gas_sphere
