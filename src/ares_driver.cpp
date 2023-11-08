//========================================================================================
// ares - a performance portable block-structured AMR compr. hydro miniapp
// Copyright (c) 2020-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon header
#include <amr_criteria/refinement_package.hpp>
#include <parthenon/parthenon.hpp>
#include <prolong_restrict/prolong_restrict.hpp>

// Plog header
#include <plog/Log.h>

// ares headers
#include "ares_driver.hpp"
#include "hydro/hydro.hpp"
#include "network/network.hpp"
#include "utils/utils.hpp"

using namespace parthenon::driver::prelude;

namespace Ares {

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;
  packages.Add(Ares::InitializeNetwork(pin.get()));
  auto network_pkg = packages.Get("Network");
  packages.Add(Ares::InitializeHydro(pin.get(), network_pkg.get()));
  return packages;
}

AresDriver::AresDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
  pin->CheckDesired("eos", "eos_type");
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection AresDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  const auto &stage_name = integrator->stage_name;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");
  auto network_pkg = blocks[0]->packages.Get("Network");

  TaskID none(0);

  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region_1 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_1[i];
    // Using "base" as u0, which already exists (and returned by using plain Get())
    auto &u0 = pmb->meshblock_data.Get();

    // Create meshblock data for register u1. This is a no-op if u1 already exists.
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);

      // init u1, see (11) in Athena++ method paper
      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            return TaskStatus::complete;
          },
          u0.get(), u1.get());
    }
  }

  // Center of Mass vector is the sum for four values: mass weighted x, y, and z,
  // total mass, and volume with density > threshold. Center of mass is then computed
  // by dividing x, y, z, by total mass, and edge radius by using the sphere volume
  // formula.
  CoM.val = {0.0, 0.0, 0.0, 0.0, 0.0};

  const int num_bins = hydro_pkg->Param<int>("num_bins");
  mass_bins.val = std::vector<Real>(num_bins);

  // This region contains one tasklist per pack, note that
  // tasklists could still be executed in parallel
  const int num_partitions = pmesh->DefaultNumPartitions();
  int reg_dep_id;
  TaskRegion &flux_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = flux_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

    reg_dep_id = 0;

    // Pass a pointer to CoM to be reduced into
    auto loc_CoM = tl.AddTask(none, utils::SumMass, mu0.get(), &CoM.val);

    // Make local reduce a regional dependency so dependent tasks can't execute until
    // all lists finish
    flux_region.AddRegionalDependencies(reg_dep_id, i, loc_CoM);
    reg_dep_id++;

    // Start a global non-blocking MPI_Iallreduce
    auto start_global_CoM =
        (i == 0 ? tl.AddTask(loc_CoM, &AllReduce<std::vector<Real>>::StartReduce, &CoM,
                             MPI_SUM)
                : none);

    auto finish_global_CoM =
        tl.AddTask(start_global_CoM, &AllReduce<std::vector<Real>>::CheckReduce, &CoM);
    flux_region.AddRegionalDependencies(reg_dep_id, i, finish_global_CoM);
    reg_dep_id++;

    // Convert CoM to [xcenter, ycenter, zcenter, total mass, edge radius] instead
    // of weighted [x total, y total, z total, total mass, total volume]
    auto simplify_CoM = tl.AddTask(
        finish_global_CoM,
        [i](std::vector<Real> *vec) {
          auto &v = *vec;
          v[0] = v[0] / v[3];
          v[1] = v[1] / v[3];
          v[2] = v[2] / v[3];
          v[4] = std::cbrt(0.75 * v[4] / M_PI);
          return TaskStatus::complete;
        },
        &CoM.val);

    auto report_CoM = (utils::ShouldLog(i) ? tl.AddTask(
                                                 simplify_CoM,
                                                 [i](std::vector<Real> *vec) {
                                                   auto &v = *vec;
                                                   PLOG(plog::debug)
                                                       << utils::TaskInfo(i)
                                                       << "Center of mass: X=" << v[0]
                                                       << " Y=" << v[1] << " Z=" << v[2]
                                                       << " Mass=" << v[3]
                                                       << " Edge Radius=" << v[4];
                                                   return TaskStatus::complete;
                                                 },
                                                 &CoM.val)
                                           : none);

    // Compute mass bins for gravity
    auto local_bin =
        tl.AddTask(simplify_CoM, utils::BinMasses, mu0.get(), &mass_bins.val, &CoM.val);
    flux_region.AddRegionalDependencies(reg_dep_id, i, local_bin);
    reg_dep_id++;

    auto start_global_bin =
        (i == 0 ? tl.AddTask(local_bin, &AllReduce<std::vector<Real>>::StartReduce,
                             &mass_bins, MPI_SUM)
                : none);

    auto finish_global_bin = tl.AddTask(
        start_global_bin, &AllReduce<std::vector<Real>>::CheckReduce, &mass_bins);
    flux_region.AddRegionalDependencies(reg_dep_id, i, finish_global_bin);
    reg_dep_id++;

    // Convert mass bins to mass enclosed by inclusive scan
    auto simplify_bin = tl.AddTask(
        finish_global_bin,
        [num_bins](std::vector<Real> *vec) {
          auto &v = *vec;
          for (int i = 1; i < num_bins; ++i) {
            v[i] += v[i - 1];
          }
          return TaskStatus::complete;
        },
        &mass_bins.val);

    auto report_bin = (utils::ShouldLog(i)
                           ? tl.AddTask(
                                 simplify_bin,
                                 [i, num_bins](std::vector<Real> *vec) {
                                   auto &v = *vec;
                                   std::string log_string;
                                   for (int j = 0; j < num_bins; ++j) {
                                     log_string += std::to_string(v[j]) + " ";
                                   }
                                   PLOG(plog::verbose) << utils::TaskInfo(i)
                                                       << "Mass enclosed: " << log_string;
                                   return TaskStatus::complete;
                                 },
                                 &mass_bins.val)
                           : none);

    const auto any = parthenon::BoundaryType::any;
    tl.AddTask(none, parthenon::cell_centered_bvars::StartReceiveBoundBufs<any>, mu0);
    tl.AddTask(none, parthenon::cell_centered_bvars::StartReceiveFluxCorrections, mu0);

    // Calculate gravity source terms
    const auto gravity_str = (stage == 1) ? "gravity_first_stage" : "gravity_other_stage";
    GravityFun_t *calc_gravity_fun = hydro_pkg->Param<GravityFun_t *>(gravity_str);

    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);

    // Solve the nuclear network
    // TODO(aholas) Make sure that the network stages make sense this way
    const auto network_str = (stage == 1) ? "network_first_stage" : "network_other_stage";
    NetworkFun_t *calc_network_fun = network_pkg->Param<NetworkFun_t *>(network_str);

    // Calculate fluxes (will be stored in the x1, x2, x3 flux arrays of each var)
    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    // Correct for fluxes across levels (to maintain conservative nature of update)
    auto send_flx = tl.AddTask(
        calc_flux, parthenon::cell_centered_bvars::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(
        calc_flux, parthenon::cell_centered_bvars::ReceiveFluxCorrections, mu0);
    auto set_flx =
        tl.AddTask(recv_flx, parthenon::cell_centered_bvars::SetFluxCorrections, mu0);

    // Compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        set_flx, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    auto calc_network = tl.AddTask(update, calc_network_fun, mu0,
                                   integrator->beta[stage - 1] * integrator->dt);
    auto calc_gravity =
        tl.AddTask(update | simplify_bin, calc_gravity_fun, mu0, &CoM.val, &mass_bins.val,
                   integrator->beta[stage - 1] * integrator->dt);

    // Update ghost cells (local and non local)
    // Note that Parthenon also support to add those tasks manually for more fine-grained
    // control.
    parthenon::cell_centered_bvars::AddBoundaryExchangeTasks(calc_gravity | calc_network,
                                                             tl, mu0, pmesh->multilevel);
  }

  TaskRegion &async_region_3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, u0);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, u0);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mu0.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Ares