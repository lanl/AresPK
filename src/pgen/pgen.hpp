#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Singularity EOS
#include <singularity-eos/eos/eos.hpp>

namespace linear_wave {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace blast

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace kh

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace sod

namespace burn_tube {
using namespace parthenon::driver::prelude;
using EOS = singularity::Variant<singularity::IdealGas, singularity::StellarCollapse>;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace burn_tube

namespace rt {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace rt

namespace gas_sphere {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace gas_sphere

namespace white_dwarf {
using namespace parthenon::driver::prelude;
Real RK4_solve(Real y_init, Real k1, Real k2, Real k3, Real k4, Real dy);
Real mass_solve(Real mass_init, Real density, Real radius, Real dr);
Real pres_funct(Real density, Real radius, Real mass);
Real pres_solve(Real pres_init, Real density, Real radius, Real mass, Real dr);
Real dens_funct(Real pressure);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace white_dwarf
#endif // PGEN_PGEN_HPP_
