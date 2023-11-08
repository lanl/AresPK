// namespace Hydro
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../gravitation/gsolvers.hpp"
#include "../main.hpp"
#include "../network/network.hpp"
#include "../pgen/pgen.hpp"
#include "../recon/plm_simple.hpp"
#include "../recon/ppm_simple.hpp"
#include "../refinement/refinement.hpp"
#include "defs.hpp"
#include "hydro.hpp"
#include "outputs/outputs.hpp"
#include "rsolvers/rsolvers.hpp"
#include "utils/error_checking.hpp"

// Singularity EOS
#include <singularity-eos/eos/eos.hpp>

using namespace parthenon::package::prelude;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::MeshData;
using parthenon::Real;
using EOS = singularity::Variant<singularity::IdealGas, singularity::StellarCollapse,
                                 singularity::Helmholtz>;

extern char DATA_DIR[PATH_MAX];

// *************************************************//
// define the "physics" package Hydro, which  *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace Ares {

using parthenon::HistoryOutputVar;

template <Hst hst, int idx = -1>
Real HydroHst(MeshData<Real> *md) {
  const auto &cellbounds = md->GetBlockData(0)->GetBlockPointer()->cellbounds;
  IndexRange ib = cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  Real sum = 0.0;

  // Sanity checks
  if ((hst == Hst::idx) && (idx < 0)) {
    PARTHENON_FAIL("Idx based hst output needs index >= 0");
  }
  Kokkos::parallel_reduce(
      "HydroHst",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        if (hst == Hst::idx) {
          lsum += cons(idx, k, j, i) * coords.CellVolume(k, j, i);
        } else if (hst == Hst::ekin) {
          lsum += 0.5 / cons(IDN, k, j, i) *
                  (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                   SQR(cons(IM3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        }
      },
      sum);

  return sum;
}

// TOOD(pgrete) check is we can enlist this with FillDerived directly
// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
template <class T>
void ConsToPrim(MeshData<Real> *md) {
  const auto &eos =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<T>("eos");
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto gamma_pack = md->PackVariables(std::vector<std::string>{"gamma"});
  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::entire);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::entire);
  auto kb = cellbounds.GetBoundsK(IndexDomain::entire);
  auto density_floor_ = pmb->packages.Get("Hydro")->Param<Real>("hydro/density_floor");
  auto pressure_floor_ = pmb->packages.Get("Hydro")->Param<Real>("hydro/pressure_floor");
  int num_species = pmb->packages.Get("Hydro")->Param<int>("num_species");
  auto eos_lambda_pack = md->PackVariables(std::vector<std::string>{"eos_lambda"});
  bool update_lambda = pmb->packages.Get("Hydro")->Param<bool>("update_lambda");
  const auto &nucdata =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Network")->Param<NucData>(
          "nucdata");

  // Temperature limits for root finding & initial guess
  static constexpr int ilTMin_ = 3;
  static constexpr int ilTMax_ = 13;
  static constexpr Real lTMin = ilTMin_;
  static constexpr Real lTMax = ilTMax_;

  pmb->par_for(
      "ConservedToPrimitive", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        auto &gamma = gamma_pack(b);
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        Real &w_d = prim(IDN, k, j, i);
        Real &w_vx = prim(IV1, k, j, i);
        Real &w_vy = prim(IV2, k, j, i);
        Real &w_vz = prim(IV3, k, j, i);
        Real &w_p = prim(IPR, k, j, i);

        Real &gamma_c = gamma(0, k, j, i);
        Real &gamma_e = gamma(1, k, j, i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ? u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0 / u_d;
        w_vx = u_m1 * di;
        w_vy = u_m2 * di;
        w_vz = u_m3 * di;

        Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));

        Real xsum = 0.0;
        for (int is = NHYDRO; is < NHYDRO + num_species; is++) {
          Real &u_s = cons(is, k, j, i);
          xsum += u_s * di;
        }
        Real temp;
        // Renormalize abundances to 1
        for (int is = NHYDRO; is < NHYDRO + num_species; is++) {
          Real &u_s = cons(is, k, j, i);
          Real &w_s = prim(is, k, j, i);
          const Real u_sc = cons(is, k, j, i);
          const Real w_sc = prim(is, k, j, i);
          w_s = u_s * di;
          u_s /= xsum;
          w_s /= xsum;
        }

        auto &eos_lambda = eos_lambda_pack(b);
        Real &abar = eos_lambda(0, k, j, i);
        Real &zbar = eos_lambda(1, k, j, i);
        Real &lT = eos_lambda(2, k, j, i);
        if (lT == 0.0) {
          lT = 7.0;
        }
        if (update_lambda) {

          Real ab = 0.0;
          Real zb = 0.0;

          for (int n = 0; n < num_species; ++n) {
            Real ymass = prim(NHYDRO + n, k, j, i) / nucdata.na[n];
            ab += ymass;
            zb += nucdata.nz[n] * ymass;
          }

          abar = xsum / ab;
          zbar = zb / xsum * abar;

          Real lambda_tmp[3] = {abar, zbar, lT};

          temp = eos.TemperatureFromDensityInternalEnergy(u_d, (u_e - e_k) / u_d,
                                                          lambda_tmp);
          lT = std::log10(temp);
          // If initial guess is outside of range, set it to default value
          if ((lT < lTMin) || (lT > lTMax)) {
            lT = 7;
          }
        }

        Real lambda[3] = {abar, zbar, lT};
        Real gm1 =
            eos.GruneisenParamFromDensityInternalEnergy(u_d, (u_e - e_k) / u_d, lambda);
        w_p = gm1 * (u_e - e_k);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k);
        w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;

        gamma_c =
            eos.BulkModulusFromDensityInternalEnergy(u_d, (u_e - e_k) / u_d, lambda) /
            w_p;
        gamma_e =
            eos.GruneisenParamFromDensityInternalEnergy(u_d, (u_e - e_k) / u_d, lambda) +
            1.0;
      });
}

std::shared_ptr<StateDescriptor>
InitializeHydro(ParameterInput *pin, parthenon::StateDescriptor *network_pkg) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  const Real dens_thresh = pin->GetOrAddReal("utils/compute_edge", "dens_thresh", 1.e-3);
  const Real newton_G = pin->GetOrAddReal("gravity", "newton_G", 1.0);
  const int num_bins = pin->GetOrAddInteger("gravity", "num_bins", 10);
  pkg->AddParam<>("cfl", cfl);
  pkg->AddParam<>("dens_thresh", dens_thresh);
  pkg->AddParam<>("num_bins", num_bins);
  pkg->AddParam<>("newton_G", newton_G);

  // Fluid type
  int nhydro = -1;
  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "euler");
  auto fluid = Fluid::undefined;
  bool calc_c_h = false; // calculate hyperbolic divergence cleaning speed

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = GetNVars<Fluid::euler>();
  } else {
    PARTHENON_FAIL("Ares: Unknown fluid method");
  }

  int num_species = network_pkg->Param<int>("num_species");
  nhydro += num_species;
  pkg->AddParam("num_species", num_species);

  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);
  pkg->AddParam<>("calc_c_h", calc_c_h);

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IDN>, "mass"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM1>, "1-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM2>, "2-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM3>, "3-mom"));
  hst_vars.emplace_back(
      HistoryOutputVar(parthenon::UserHistoryOperation::sum, HydroHst<Hst::ekin>, "KE"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IEN>, "tot-E"));
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  // not using GetOrAdd here until there's a reasonable default
  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < 2) {
    PARTHENON_FAIL("Ares hydro: Need more ghost zones for chosen reconstruction.");
  }

  Real dfloor = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * float_min));
  Real pfloor = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024 * float_min));

  pkg->AddParam<Real>("hydro/density_floor", dfloor);
  pkg->AddParam<Real>("hydro/pressure_floor", pfloor);

  const auto eos_str = pin->GetOrAddString("eos", "eos_type", "ideal");
  if (eos_str == "ideal") {
    const Real gm1_in = pin->GetOrAddReal("eos", "gm1", 0.6666667);
    const Real cv_in = pin->GetOrAddReal("eos", "Cv", 1.5);
    singularity::EOS eos = singularity::IdealGas(gm1_in, cv_in);
    singularity::EOS eos_device = eos.GetOnDevice();
    pkg->AddParam<>("eos", eos_device);
    pkg->AddParam<>("eos_host", eos);
    pkg->AddParam<>("update_lambda", false);
  } else if (eos_str == "helm") {
    // Check if species are present
    // TODO (aholas) Make this check more robust
    if (num_species < 1) {
      PARTHENON_FAIL("Ares: Helmholtz equation of state requires at least on species.")
    }
    const bool eos_rad = pin->GetOrAddReal("eos", "rad", 1);
    const bool eos_gas = pin->GetOrAddReal("eos", "gas", 1);
    const bool eos_coulomb = pin->GetOrAddReal("eos", "coulomb", 1);
    const bool eos_ionized = pin->GetOrAddReal("eos", "ionized", 1);
    const bool eos_degenerate = pin->GetOrAddReal("eos", "degenerate", 1);
    std::string helm_table_name =
        pin->GetOrAddString("eos", "helm_table", "helm_table.dat");
    const auto helm_table_file = DATA_DIR + helm_table_name;
    singularity::EOS eos = singularity::Helmholtz(
        helm_table_file, eos_rad, eos_gas, eos_coulomb, eos_ionized, eos_degenerate);
    singularity::EOS eos_device = eos.GetOnDevice();
    pkg->AddParam<>("eos", eos_device);
    pkg->AddParam<>("eos_host", eos);
    pkg->AddParam<>("update_lambda", true);
  } else {
    PARTHENON_FAIL("Ares: Unknown equation of state.")
  }
  std::string field_name = "eos_lambda";
  std::vector<std::string> eos_lambda_labels(3);
  eos_lambda_labels[0] = "Abar";
  eos_lambda_labels[1] = "Zbar";
  eos_lambda_labels[2] = "lT";
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::Intensive},
             std::vector<int>({3}), eos_lambda_labels);
  pkg->AddField(field_name, m);

  pkg->FillDerivedMesh = ConsToPrim<singularity::EOS>;

  pkg->EstimateTimestepMesh = EstimateTimestep;

  auto network_enabled = network_pkg->Param<bool>("network_enabled");
  pkg->AddParam("network_enabled", network_enabled);

  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 0);
  if (network_enabled && (scratch_level == 0)) {
    PARTHENON_WARN(
        "Ares: scratch level 0 should not be used with nuclear network enabled!")
  }
  pkg->AddParam("scratch_level", scratch_level);

  field_name = "cons";
  std::vector<std::string> cons_labels(nhydro);
  cons_labels[IDN] = "Density";
  cons_labels[IM1] = "MomentumDensity1";
  cons_labels[IM2] = "MomentumDensity2";
  cons_labels[IM3] = "MomentumDensity3";
  cons_labels[IEN] = "TotalEnergyDensity";
  if (network_enabled) {
    for (int i = NHYDRO; i < nhydro; i++) {
      auto label = "rhoxnuc" + std::to_string(i - NHYDRO);
      cons_labels[i] = label;
    }
  }
  m = Metadata(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro}), cons_labels);
  pkg->AddField(field_name, m);

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
  field_name = "prim";
  std::vector<std::string> prim_labels(nhydro);
  prim_labels[IDN] = "Density";
  prim_labels[IV1] = "Velocity1";
  prim_labels[IV2] = "Velocity2";
  prim_labels[IV3] = "Velocity3";
  prim_labels[IPR] = "Pressure";
  if (network_enabled) {
    for (int i = NHYDRO; i < nhydro; i++) {
      auto label = "xnuc" + std::to_string(i - NHYDRO);
      prim_labels[i] = label;
    }
  }
  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro}),
               prim_labels);
  pkg->AddField(field_name, m);

  field_name = "gamma";
  std::vector<std::string> gamma_labels(2);
  gamma_labels[0] = "gamma_c";
  gamma_labels[1] = "gamma_e";
  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({2}), gamma_labels);
  pkg->AddField(field_name, m);

  const auto refine_str = pin->GetOrAddString("refinement", "type", "unset");
  if (refine_str == "pressure_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::PressureGradient;
    const auto thr = pin->GetOrAddReal("refinement", "threshold_pressure_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_pressure_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_pressure_gradient", thr);
  } else if (refine_str == "xyvelocity_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::VelocityGradient;
    const auto thr =
        pin->GetOrAddReal("refinement", "threshold_xyvelosity_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_xyvelocity_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_xyvelocity_gradient", thr);
  } else if (refine_str == "maxdensity") {
    pkg->CheckRefinementBlock = refinement::other::MaxDensity;
    const auto deref_below =
        pin->GetOrAddReal("refinement", "maxdensity_deref_below", 0.0);
    const auto refine_above =
        pin->GetOrAddReal("refinement", "maxdensity_refine_above", 0.0);
    PARTHENON_REQUIRE(deref_below > 0.,
                      "Make sure to set refinement/maxdensity_deref_below > 0.");
    PARTHENON_REQUIRE(refine_above > 0.,
                      "Make sure to set refinement/maxdensity_refine_above > 0.");
    PARTHENON_REQUIRE(deref_below < refine_above,
                      "Make sure to set refinement/maxdensity_deref_below < "
                      "refinement/maxdensity_refine_above");
    pkg->AddParam<Real>("refinement/maxdensity_deref_below", deref_below);
    pkg->AddParam<Real>("refinement/maxdensity_refine_above", refine_above);
  }

  // Select gravity solver
  const auto gravity_str = pin->GetOrAddString("gravity", "gravity_solver", "none");
  auto gravity = GravitySolver::undefined;
  if (gravity_str == "monopole") {
    gravity = GravitySolver::monopole;
  } else if (gravity_str == "none") {
    gravity = GravitySolver::none;
  } else if (gravity_str == "constant") {
    gravity = GravitySolver::constant;
  } else {
    PARTHENON_FAIL("Ares: Unknown gravity solver.");
  }

  const Real grav_accel =
      pin->GetOrAddReal("gravity", "grav_accel", 980.0); // CGS Gravity
  pkg->AddParam<Real>("grav_accel", grav_accel);
  pkg->AddParam<>("gravity", gravity);

  // Map contaning all compiled in gravity functions
  std::map<GravitySolver, GravityFun_t *> gravity_functions{};

  add_gravity_fun<GravitySolver::constant>(gravity_functions);
  add_gravity_fun<GravitySolver::monopole>(gravity_functions);
  add_gravity_fun<GravitySolver::none>(gravity_functions);

  // Gravity used in all stages expect the first. First stage is set below based on
  // integr.
  GravityFun_t *gravity_other_stage = nullptr;
  gravity_other_stage = gravity_functions.at(gravity);

  GravityFun_t *gravity_first_stage = gravity_other_stage;
  pkg->AddParam<GravityFun_t *>("gravity_first_stage", gravity_first_stage);
  pkg->AddParam<GravityFun_t *>("gravity_other_stage", gravity_other_stage);

  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;

  if (recon_str == "plm") {
    recon = Reconstruction::plm;
    recon_need_nghost = 2;
  } else if (recon_str == "ppm") {
    recon = Reconstruction::ppm;
    recon_need_nghost = 3;
  } else {
    PARTHENON_FAIL("Ares hydro: Unknown reconstruction method.");
  }

  // Adding recon independently of flux function pointer as it's used in 3D flux func.
  pkg->AddParam<>("reconstruction", recon);

  // Use hyperbolic timestep constraint by default
  bool calc_dt_hyp = true;
  const auto riemann_str = pin->GetOrAddString("hydro", "riemann", "hllc");
  auto riemann = RiemannSolver::undefined;
  if (riemann_str == "hlle") {
    riemann = RiemannSolver::hlle;
  } else if (riemann_str == "hllc") {
    riemann = RiemannSolver::hllc;
  } else {
    PARTHENON_FAIL("Ares: Unknown Riemann solver.")
  }
  pkg->AddParam<>("riemann", riemann);

  // Get network solver
  auto network_solver = network_pkg->Param<NetworkSolver>("network_solver");
  pkg->AddParam<>("network_solver", network_solver);

  // Map contaning all compiled in flux functions
  std::map<std::tuple<Fluid, Reconstruction, RiemannSolver, NetworkSolver>, FluxFun_t *>
      flux_functions{};
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hlle,
               NetworkSolver::none>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hlle,
               NetworkSolver::nse>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hllc,
               NetworkSolver::none>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hllc,
               NetworkSolver::nse>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hlle,
               NetworkSolver::none>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hlle,
               NetworkSolver::nse>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hllc,
               NetworkSolver::none>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hllc,
               NetworkSolver::nse>(flux_functions);

  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage = nullptr;
  flux_other_stage =
      flux_functions.at(std::make_tuple(fluid, recon, riemann, network_solver));

  FluxFun_t *flux_first_stage = flux_other_stage;
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  if (nghost < recon_need_nghost) {
    PARTHENON_FAIL("Ares hydro: Need more ghost zones for chosen reconstruction.");
  }

  return pkg;
}

Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  static constexpr Real C_LIGHT = 2.99792458e10;
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &eos_lambda_pack = md->PackVariables(std::vector<std::string>{"eos_lambda"});
  const auto &eos = hydro_pkg->Param<singularity::EOS>("eos");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &cons = cons_pack(b);
        auto &eos_lambda = eos_lambda_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        Real w[(NHYDRO)];
        w[IDN] = prim(IDN, k, j, i);
        w[IV1] = prim(IV1, k, j, i);
        w[IV2] = prim(IV2, k, j, i);
        w[IV3] = prim(IV3, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real lambda_max_x, lambda_max_y, lambda_max_z;
        Real e_internal = (cons(IEN, k, j, i) -
                           0.5 * w[IDN] * (SQR(w[IV1]) + SQR(w[IV2]) + SQR(w[IV3]))) /
                          w[IDN];

        Real &abar = eos_lambda(0, k, j, i);
        Real &zbar = eos_lambda(1, k, j, i);
        Real &lT = eos_lambda(2, k, j, i);
        Real lambda[3] = {abar, zbar, lT};
        Real bulkmod =
            eos.BulkModulusFromDensityInternalEnergy(w[IDN], e_internal, lambda);

        const Real gamma_c = bulkmod / w[IPR];
        lambda_max_x =
            C_LIGHT *
            std::sqrt(gamma_c / (1 + (e_internal + SQR(C_LIGHT)) * w[IDN] / w[IPR]));
        lambda_max_y = lambda_max_x;
        lambda_max_z = lambda_max_x;

        min_dt = fmin(min_dt, coords.Dxc<1>(k, j, i) / (fabs(w[IV1]) + lambda_max_x));
        if (nx2) {
          min_dt = fmin(min_dt, coords.Dxc<2>(k, j, i) / (fabs(w[IV2]) + lambda_max_y));
        }
        if (nx3) {
          min_dt = fmin(min_dt, coords.Dxc<3>(k, j, i) / (fabs(w[IV3]) + lambda_max_z));
        }
      },
      Kokkos::Min<Real>(min_dt_hyperbolic));

  return cfl_hyp * min_dt_hyperbolic;
}

// Calculate gravity
template <GravitySolver gsolver>
TaskStatus CalculateGravity(std::shared_ptr<MeshData<Real>> &md, std::vector<Real> *CoM,
                            std::vector<Real> *mass_bins, Real dt) {
  auto gravity = gravitation::Gravitation<gsolver>();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  auto const &prim_pack = md->PackVariables(std::vector<std::string>({"prim"}));
  auto const &cons_pack = md->PackVariables(std::vector<std::string>({"cons"}));

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  if constexpr (gsolver == GravitySolver::constant)
    gravity.set_parameters(hydro_pkg->Param<Real>("grav_accel"));

  if constexpr (gsolver == GravitySolver::monopole)
    gravity.set_parameters(CoM, mass_bins, hydro_pkg->Param<Real>("newton_G"));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "UpdateGravity", DevExecSpace(), 0, cons_pack.GetDim(5) - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons.GetCoords(b);
        gravity.Solve(prim, cons, coords, i, j, k, dt);
      });

  return TaskStatus::complete;
} // CalculateGravity

// Calculate fluxes using scratch pad memory, i.e. over cached pencils in i-dir.
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver, NetworkSolver nsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (pmb->block_size.nx2 > 1) {
    if (pmb->block_size.nx3 == 1) // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    else // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
  }

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");
  const int num_species = pkg->Param<int>("num_species");

  const auto &eos = pkg->Param<singularity::EOS>("eos");

  auto num_scratch_vars = nhydro;
  auto prim_list = std::vector<std::string>({"prim"});
  auto gamma_list = std::vector<std::string>({"gamma"});

  auto const &prim_pack = md->PackVariables(prim_list);
  auto const &gamma_pack = md->PackVariables(gamma_list);

  const auto &eos_lambda_pack = md->PackVariables(std::vector<std::string>{"eos_lambda"});

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 4;

  auto riemann = Riemann<fluid, rsolver, nsolver>();

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "x1 flux", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_pack.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = cons_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const auto &gamma = gamma_pack(b);
        auto &eos_lambda = eos_lambda_pack(b);
        auto &cons = cons_pack(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level), 2, nx1);
        parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level), 2, nx1);
        // get reconstructed state on faces
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, gamma, ifl, ifr);

        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        riemann.Solve(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, ifl, ifr, eos,
                      num_species, eos_lambda);
      });
  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 6;
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (pmb->block_size.nx3 == 1) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x2 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_pack.GetDim(5) - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          const auto &coords = cons_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto &gamma = gamma_pack(b);
          auto &eos_lambda = eos_lambda_pack(b);
          auto &cons = cons_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level), 2, nx1);
          parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level), 2, nx1);
          parthenon::ScratchPad2D<Real> iflb(member.team_scratch(scratch_level), 2, nx1);
          for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, prim, wlb, wr);
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, gamma, iflb, ifr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV2, wl, wr, cons, ifl, ifr, eos,
                            num_species, eos_lambda);
              member.team_barrier();
            }

            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
            auto *tmpf = ifl.data();
            ifl.assign_data(iflb.data());
            iflb.assign_data(tmpf);
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_pack.GetDim(5) - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
          const auto &coords = cons_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto &gamma = gamma_pack(b);
          auto &eos_lambda = eos_lambda_pack(b);
          auto &cons = cons_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> ifl(member.team_scratch(scratch_level), 2, nx1);
          parthenon::ScratchPad2D<Real> ifr(member.team_scratch(scratch_level), 2, nx1);
          parthenon::ScratchPad2D<Real> iflb(member.team_scratch(scratch_level), 2, nx1);
          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, prim, wlb, wr);
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, gamma, iflb, ifr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV3, wl, wr, cons, ifl, ifr, eos,
                            num_species, eos_lambda);
              member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
            auto *tmpf = ifl.data();
            ifl.assign_data(iflb.data());
            iflb.assign_data(tmpf);
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace Ares
