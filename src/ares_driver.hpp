#ifndef ARES_ARES_DRIVER_HPP_
#define ARES_ARES_DRIVER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon header
#include <parthenon/parthenon.hpp>

#include "main.hpp"

using namespace parthenon::driver::prelude;

namespace Ares {

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);

class AresDriver : public MultiStageDriver {
 public:
  AresDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteBlockTasks (driver.hpp)
  //         AdvectionDriver::MakeTaskList (advection.cpp)
  auto MakeTaskCollection(BlockList_t &blocks, int stage) -> TaskCollection;

  AllReduce<std::vector<Real>> CoM;
  AllReduce<std::vector<Real>> mass_bins;
};

} // namespace Ares

#endif // ARES_ARES_DRIVER_HPP_
