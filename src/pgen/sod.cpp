
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

using namespace parthenon::package::prelude;

namespace sod {

// TODO(pgrete) need to make this more flexible especially for other problems
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rho_l = pin->GetOrAddReal("problem/sod", "rho_l", 1.0);
  Real pres_l = pin->GetOrAddReal("problem/sod", "pres_l", 1.0);
  Real u_l = pin->GetOrAddReal("problem/sod", "u_l", 0.0);
  Real rho_r = pin->GetOrAddReal("problem/sod", "rho_r", 0.125);
  Real pres_r = pin->GetOrAddReal("problem/sod", "pres_r", 0.1);
  Real u_r = pin->GetOrAddReal("problem/sod", "u_r", 0.0);
  Real x_discont = pin->GetOrAddReal("problem/sod", "x_discont", 0.5);

  // TODO(pgrete): need to make sure an EOS is used here
  Real gamma = pin->GetReal("hydro", "gamma");

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
        if (coords.Xc<1>(k, j, i) < x_discont) {
          u(IDN, k, j, i) = rho_l;
          u(IV1, k, j, i) = rho_l * u_l;
          u(IV2, k, j, i) = 0.0;
          u(IV3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 0.5 * rho_l * u_l * u_l + pres_l / (gamma - 1.0);
        } else {
          u(IDN, k, j, i) = rho_r;
          u(IV1, k, j, i) = rho_r * u_r;
          u(IV2, k, j, i) = 0.0;
          u(IV3, k, j, i) = 0.0;
          u(IEN, k, j, i) = 0.5 * rho_r * u_r * u_r + pres_r / (gamma - 1.0);
        }
      });
}

} // namespace sod
