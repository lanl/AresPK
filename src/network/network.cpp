#include <iostream>
#include <sstream>

// Parthenon headers
#include <parthenon/package.hpp>

// Ares headers
#include "../hydro/hydro.hpp"
#include "../main.hpp"
#include "../utils/utils.hpp"
#include "network.hpp"
#include "nsolvers/nsolvers.hpp"

// Singularity EOS
#include <singularity-eos/eos/eos.hpp>

// Plog
#include <plog/Log.h>

using namespace parthenon::package::prelude;
using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::MeshData;
using parthenon::Real;
using EOS = singularity::Variant<singularity::IdealGas, singularity::StellarCollapse,
                                 singularity::Helmholtz>;

extern char DATA_DIR[PATH_MAX];

namespace Ares {

std::shared_ptr<StateDescriptor> InitializeNetwork(ParameterInput *pin) {

  const Real BOLTZMAN_CONSTANT = 1.3806504e-16;     /* g cm^2 / K s^2 */
  const Real PLANCKS_CONSTANT_H = 6.62606896e-27;   /* g cm^2 / s */
  const Real SPEED_OF_LIGHT = 2.99792458e10;        /* cm / s */
  const Real NUM_AVOGADRO = 6.02214199e23;          /* 1 / mol */
  const Real UNIFIED_ATOMIC_MASS = 1.660538782e-24; /* Conversion from amu to gram*/

  const static Real conv =
      1.602177e-12 * 1.0e3 * NUM_AVOGADRO; /* eV2erg * 1.0e3 [keV] * avogadro */

  auto pkg = std::make_shared<StateDescriptor>("Network");

  /* Load network type. If none is found it deactivates the network. */
  const auto network_str = pin->GetOrAddString("network", "solver", "none");
  bool network_enabled;
  auto network_solver = NetworkSolver::undefined;
  if (network_str == "nse") {
    network_solver = NetworkSolver::nse;
    network_enabled = true;
  } else if (network_str == "none") {
    network_solver = NetworkSolver::none;
    network_enabled = false;
  } else {
    PARTHENON_FAIL("Ares: Unknown Network solver")
  }
  pkg->AddParam("network_solver", network_solver);
  pkg->AddParam("network_enabled", network_enabled);

  // TODO(aholas) Make sure that the network stages make sense this way
  // Map containing all compile in network functions
  std::map<NetworkSolver, NetworkFun_t *> network_functions{};
  add_network_fun<NetworkSolver::nse>(network_functions);
  add_network_fun<NetworkSolver::none>(network_functions);

  // Network used in all stages except the first. First stage is set below based on
  // integr.
  NetworkFun_t *network_other_stage = nullptr;
  network_other_stage = network_functions.at(network_solver);

  NetworkFun_t *network_first_stage = network_other_stage;
  pkg->AddParam<NetworkFun_t *>("network_first_stage", network_first_stage);
  pkg->AddParam<NetworkFun_t *>("network_other_stage", network_other_stage);

  Real min_network_temp = pin->GetOrAddReal("network", "MinNetworkTemp", 3e9);
  pkg->AddParam("min_network_temp", min_network_temp);

  NucData nucdata;

  /* Check if network is enabled. Everything below this comment should only
    be required by nsolvers other than 'none', i.e. when the network is
    enabled. pkg parameters added above are needed as dummy variables even
    in cases where the network is disabled. */
  if (!network_enabled) {
    PLOG(plog::info) << "Running without nuclear network";
    pkg->AddParam<int>("num_species", 0);
    pkg->AddParam<>("nucdata", nucdata);
    return pkg;
  } else {
    PLOG(plog::info) << "Running with nuclear network";
  }

  pkg->EstimateTimestepMesh = EstimateTimestep;

  /* Load a species file. Default is loaded from data directoy,
   otherwise path has to be fiven*/
  const auto species_file_name =
      pin->GetOrAddString("network", "species", "species55.txt");
  const auto species_file_str = DATA_DIR + species_file_name;

  std::fstream species_file;
  species_file.open(species_file_str, std::ios::in);
  if (!species_file) {
    PARTHENON_FAIL("Ares: Cannot find specified species file")
  } else {
    PLOG(plog::info) << "Running network using species file " << species_file_name;
  }
  int num_species = 0;

  std::string num_spec_str;
  getline(species_file, num_spec_str);
  std::istringstream in(num_spec_str);
  in >> num_species;
  // TODO(aholas) Make this more dynamic
  if (num_species > NXNUC) {
    PARTHENON_FAIL("Ares: Exceeded the maximum number of supported species (55);")
  }
  pkg->AddParam<int>("num_species", num_species);
  PLOG(plog::info) << "Found " << num_species << " species in species file";

  std::string spec_file_instr;

  // Readin species data
  for (int i = 0; i < num_species; ++i) {
    getline(species_file, spec_file_instr);
    std::istringstream in(spec_file_instr);
    in >> nucdata.spec_name[i];
    in >> nucdata.na[i];
    in >> nucdata.nz[i];
    nucdata.nn[i] = nucdata.na[i] - nucdata.nz[i];
    PLOG(plog::verbose) << "Found " << nucdata.spec_name[i] << ", A = " << nucdata.na[i]
                        << ", Z = " << nucdata.nz[i] << " in species file";
  }
  species_file.close();

  /* Reading in the mass file to get the mass excess for each isotope in the network*/
  const auto mass_file_name = pin->GetOrAddString("network", "mass", "mass.txt");
  const auto mass_file_str = DATA_DIR + mass_file_name;

  std::fstream mass_file;
  mass_file.open(mass_file_str, std::ios::in);
  if (!mass_file) {
    PARTHENON_FAIL("Ares: Cannot find specified mass file")
  } else {
    PLOG(plog::info) << "Running network using mass file " << mass_file_name;
  }

  std::string dummy;
  for (int i = 0; i < 39; ++i) {
    getline(mass_file, dummy);
  }

  std::string mass_file_instr;
  while (mass_file.peek() != EOF) {
    getline(mass_file, mass_file_instr);
    int nz, na;
    sscanf(&mass_file_instr[10], "%d%d", &nz, &na);
    for (int i = 0; i < num_species; ++i) {
      if (nucdata.na[i] == na && nucdata.nz[i] == nz) {
        Real m_ex, q, m_mol1, m_mol2;
        sscanf(&mass_file_instr[29], "%lf", &m_ex);
        nucdata.m_ex[i] = m_ex;
        sscanf(&mass_file_instr[54], "%lf", &q); // binding energy per nucleon
        nucdata.q[i] = q * 1.602177e-12 * 1.0e3 *
                       nucdata.na[i]; /* values are in KeV / nucleon, converting to erg */
        // Convert the mass excess to mass of th nucleus
        nucdata.m[i] =
            nucdata.na[i] * UNIFIED_ATOMIC_MASS +
            nucdata.m_ex[i] * conv / NUM_AVOGADRO / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);
        sscanf(&mass_file_instr[95], "%lf%lf", &m_mol1, &m_mol2);
        nucdata.m_mol[i] = m_mol1 + m_mol2 / 1e6;
      }
    }
  }
  /* Reading in the part file to get the partition function and spin for each isotope in
   * the network*/
  const auto part_file_name = pin->GetOrAddString("network", "part", "part.txt");
  const auto part_file_str = DATA_DIR + part_file_name;

  std::fstream part_file;
  part_file.open(part_file_str, std::ios::in);
  if (!part_file) {
    PARTHENON_FAIL("Ares: Cannot find specified partition function file")
  } else {
    PLOG(plog::info) << "Running network using partition function file "
                     << part_file_name;
  }

  for (int i = 0; i < 4; ++i) {
    getline(part_file, dummy);
  }

  std::string part_file_instr;
  int count = 0;
  int count2 = 0;
  while (part_file.peek() != EOF) {
    count2++;
    getline(part_file, dummy);
    getline(part_file, part_file_instr);
    std::istringstream in(part_file_instr);
    int nz, na;
    in >> nz;
    in >> na;
    bool skipped = true;
    for (int i = 0; i < num_species; ++i) {
      if (nucdata.na[i] == na && nucdata.nz[i] == nz) {
        skipped = false;
        in >> nucdata.spin[i];
        count++;
        for (int j = 0; j < 3; ++j) {
          getline(part_file, part_file_instr);
          std::istringstream in(part_file_instr);
          for (int k = 0; k < 8; ++k) {
            in >> nucdata.part[i][j * 8 + k];
          }
        }
      }
    }
    if (skipped) {
      for (int j = 0; j < 3; ++j) {
        getline(part_file, dummy);
      }
    }
  }
  for (int i = 0; i < num_species; ++i) {
    PLOG(plog::verbose) << "Found " << nucdata.spec_name[i] << ", A = " << nucdata.na[i]
                        << ", Z = " << nucdata.nz[i] << ", M_ex = " << nucdata.m_ex[i]
                        << ", M = " << nucdata.m[i] << ", spin = " << nucdata.spin[i]
                        << ", q = " << nucdata.q[i]
                        << ", molar mass = " << nucdata.m_mol[i];
  }

  pkg->AddParam("nucdata", nucdata);
  return pkg;
}

template <NetworkSolver nsolver>
TaskStatus CalculateNetwork(std::shared_ptr<MeshData<Real>> &md, const Real dt) {

  auto network = Network<nsolver>();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Network");
  const auto &eos = md->GetBlockData(0)
                        ->GetBlockPointer()
                        ->packages.Get("Hydro")
                        ->Param<singularity::EOS>("eos");
  auto nucdata = pkg->Param<NucData>("nucdata");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &eos_lambda_pack = md->PackVariables(std::vector<std::string>{"eos_lambda"});

  const auto &cellbounds = pmb->cellbounds;
  auto ib = cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = cellbounds.GetBoundsK(IndexDomain::interior);

  auto density_floor_ = pmb->packages.Get("Hydro")->Param<Real>("hydro/density_floor");
  auto pressure_floor_ = pmb->packages.Get("Hydro")->Param<Real>("hydro/pressure_floor");
  auto nhydro = pmb->packages.Get("Hydro")->Param<int>("nhydro");

  Real min_network_temp = pkg->Param<Real>("min_network_temp");

  // Number of cells for which the network needs to be run
  int num_network_cells = 0;

  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "NetworkCells", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, int &count) {
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        auto &eos_lambda = eos_lambda_pack(b);
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        u_d = (u_d > density_floor_) ? u_d : density_floor_;
        Real e_internal = (u_e - 0.5 / u_d * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3))) / u_d;

        Real &abar = eos_lambda(0, k, j, i);
        Real &zbar = eos_lambda(1, k, j, i);
        Real &lT = eos_lambda(2, k, j, i);
        Real lambda[3] = {abar, zbar, lT};
        Real temp = eos.TemperatureFromDensityInternalEnergy(u_d, e_internal, lambda);
        if (temp >= min_network_temp) {
          Real ye = zbar / abar;
          network.Solve(temp, u_d, ye, cons, nucdata, i, j, k, dt);
          count++;
        }
      },
      Kokkos::Sum<int>(num_network_cells));

  PLOG(plog::verbose) << utils::TaskInfo() << "Running network for " << num_network_cells
                      << " cells";

  /* Step 1: Get all the cells for which the network needs to be run */

  return TaskStatus::complete;
}

} // namespace Ares
