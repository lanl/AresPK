// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

// Parthenon headers
#include "main.hpp"
#include "globals.hpp"
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "ares_driver.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

// Plog headers
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

char DATA_DIR[PATH_MAX];

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  strncpy(DATA_DIR, argv[0], sizeof(DATA_DIR));
  dirname(DATA_DIR);
  strncat(DATA_DIR, "/../data/", sizeof(DATA_DIR) - 1);

  auto log_level = plog::info;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--log-level") {
      if (i + 1 < argc) {
        log_level = plog::severityFromString(argv[++i]);
        if (log_level == plog::none) {
          std::cerr << "--log-level option requires an argument from [V=verbose, "
                       "D=debug, I=info, W=warning, E=error, F=fatal]"
                    << std::endl;
          return 1;
        }
      } else {
        std::cerr << "--log-level option requires an argument from [V=verbose, D=debug, "
                     "I=info, W=warning, E=error, F=fatal]"
                  << std::endl;
        return 1;
      }
    }
  }

  static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
  plog::init(log_level, &consoleAppender);

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Redefine defaults
  pman.app_input->ProcessPackages = Ares::ProcessPackages;
  const auto problem = pman.pinput->GetOrAddString("job", "problem_id", "unset");
  if (problem == "linear_wave") {
    pman.app_input->InitUserMeshData = linear_wave::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop;
  } else if (problem == "blast") {
    pman.app_input->ProblemGenerator = blast::ProblemGenerator;
  } else if (problem == "kh") {
    pman.app_input->ProblemGenerator = kh::ProblemGenerator;
  } else if (problem == "rt") {
    pman.app_input->ProblemGenerator = rt::ProblemGenerator;
  } else if (problem == "gas_sphere") {
    pman.app_input->ProblemGenerator = gas_sphere::ProblemGenerator;
  } else if (problem == "sod") {
    pman.app_input->ProblemGenerator = sod::ProblemGenerator;
  } else if (problem == "burn_tube") {
    pman.app_input->ProblemGenerator = burn_tube::ProblemGenerator;
  } else if (problem == "white_dwarf") {
    pman.app_input->ProblemGenerator = white_dwarf::ProblemGenerator;
  } else if (problem == "unset") {
    PARTHENON_FAIL("Ares: Problem unset. Please specify a problem.")
  } else {
    PARTHENON_FAIL("Ares: Problem unknown. Please specify a valid problem.")
  }
  pman.ParthenonInitPackagesAndMesh();

  // Startup the corresponding driver for the integrator
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting up Ares driver" << std::endl;
  }

  Ares::AresDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
