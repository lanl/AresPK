
// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// Athena headers
#include "../main.hpp"
#include "parthenon/prelude.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

// Singularity EOS
#include <singularity-eos/eos/eos.hpp>

using namespace parthenon::package::prelude;
using EOS = singularity::Variant<singularity::IdealGas, singularity::StellarCollapse,
                                 singularity::Helmholtz>;

extern char DATA_DIR[PATH_MAX];

namespace burn_tube {

// TODO(pgrete) need to make this more flexible especially for other problems
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rho_l = pin->GetOrAddReal("problem/burn_tube", "rho_l", 1.0);
  Real pres_l = pin->GetOrAddReal("problem/burn_tube", "pres_l", 1.0);
  Real u_l = pin->GetOrAddReal("problem/burn_tube", "u_l", 0.0);
  Real rho_r = pin->GetOrAddReal("problem/burn_tube", "rho_r", 1.0);
  Real pres_r = pin->GetOrAddReal("problem/burn_tube", "pres_r", 1.0);
  Real u_r = pin->GetOrAddReal("problem/burn_tube", "u_r", 0.0);
  Real x_ign = pin->GetOrAddReal("problem/burn_tube", "x_ign", 0.5);
  Real x_ign_width = pin->GetOrAddReal("problem/burn_tube", "x_width", 0.0125);
  Real temp_ign = pin->GetOrAddReal("problem/burn_tube", "temp_ign", 7e9);
  Real temp_background = pin->GetOrAddReal("problem/burn_tube", "temp_background", 1e8);

  const auto &eos = pmb->packages.Get("Hydro")->Param<singularity::EOS>("eos");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;
  // setup uniform ambient medium with spherical over-pressured region
  pmb->par_for(
      "ProblemGenerator Blast", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // Hardcoded abar and zbar values
        Real lambda_ign[3] = {13.714285714285715, 6.857142857142858,
                              std::log10(temp_ign)};
        Real lambda_bcg[3] = {4.0, 2.0, std::log10(temp_background)};
        if (coords.Xc<1>(k, j, i) < x_ign + x_ign_width &&
            coords.Xc<1>(k, j, i) > x_ign - x_ign_width) {
          u(IDN, k, j, i) = rho_l;
          u(IV1, k, j, i) = rho_l * u_l;
          u(IV2, k, j, i) = 0.0;
          u(IV3, k, j, i) = 0.0;
          u(IEN, k, j, i) =
              eos.InternalEnergyFromDensityTemperature(rho_l, temp_ign, lambda_ign) *
                  rho_l +
              0.5 / rho_l * SQR(u(IV1, k, j, i));
          // 50/50 CO mix in blast
          for (int is = NHYDRO; is < NHYDRO + NXNUC; is++) {
            u(is, k, j, i) = 0.0;
          }
          u(NHYDRO + 4, k, j, i) = rho_l * 0.5;
          u(NHYDRO + 10, k, j, i) = rho_l * 0.5;
        } else {
          u(IDN, k, j, i) = rho_r;
          u(IV1, k, j, i) = rho_r * u_r;
          u(IV2, k, j, i) = 0.0;
          u(IV3, k, j, i) = 0.0;
          u(IEN, k, j, i) = eos.InternalEnergyFromDensityTemperature(
                                rho_r, temp_background, lambda_bcg) *
                                rho_r +
                            0.5 / rho_r * SQR(u(IV1, k, j, i));
          // He background
          for (int is = NHYDRO; is < NHYDRO + NXNUC; is++) {
            u(is, k, j, i) = 0.0;
          }
          u(NHYDRO + 2, k, j, i) = rho_r * 1.0;
        }
      });
}

} // namespace burn_tube
