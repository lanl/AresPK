#ifndef ARES_HYDRO_HPP_
#define ARES_HYDRO_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

#include "../main.hpp"

using namespace parthenon::package::prelude;

namespace Ares {

template <class T>
void ConsToPrim(MeshData<Real> *md);

std::shared_ptr<StateDescriptor> InitializeHydro(ParameterInput *pin,
                                                 parthenon::StateDescriptor *hydro_pkg);

Real EstimateTimestep(MeshData<Real> *md);

template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver, NetworkSolver nsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md);
using FluxFun_t = decltype(CalculateFluxes<Fluid::euler, Reconstruction::plm,
                                           RiemannSolver::hlle, NetworkSolver::none>);
using FluxFunKey_t = std::tuple<Fluid, Reconstruction, RiemannSolver, NetworkSolver>;

// Add flux function pointer to map containing all compiled in flux functions
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver, NetworkSolver nsolver>
void add_flux_fun(std::map<FluxFunKey_t, FluxFun_t *> &flux_functions) {
  flux_functions[std::make_tuple(fluid, recon, rsolver, nsolver)] =
      Ares::CalculateFluxes<fluid, recon, rsolver, nsolver>;
}

template <GravitySolver gsolver>
TaskStatus CalculateGravity(std::shared_ptr<MeshData<Real>> &md, std::vector<Real> *CoM,
                            std::vector<Real> *mass_bins, Real dt);
using GravityFun_t = decltype(CalculateGravity<GravitySolver::monopole>);

using GravityFunKey_t = GravitySolver;

// Add flux function pointer to map containing all compiled in flux functions
template <GravitySolver gsolver>
void add_gravity_fun(std::map<GravityFunKey_t, GravityFun_t *> &gravity_functions) {
  gravity_functions[gsolver] = Ares::CalculateGravity<gsolver>;
}

// Get number of "fluid" variable used
template <Fluid fluid>
constexpr size_t GetNVars();

template <>
constexpr size_t GetNVars<Fluid::euler>() {
  return 5; // rho, u_x, u_y, u_z, E
}

} // namespace Ares

#endif // ARES_HYDRO_HPP_
