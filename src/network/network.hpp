#ifndef ARES_NETWORK_HPP_
#define ARES_NETWORK_HPP_

#include <parthenon/package.hpp>

#include "../main.hpp"

using namespace parthenon::package::prelude;

namespace Ares {

std::shared_ptr<StateDescriptor> InitializeNetwork(ParameterInput *pin);

template <NetworkSolver nsolver>
TaskStatus CalculateNetwork(std::shared_ptr<MeshData<Real>> &md, const Real dt);
using NetworkFun_t = decltype(CalculateNetwork<NetworkSolver::nse>);

using NetworkFunKey_t = NetworkSolver;

// Add flux function pointer to map containing all compiled in flux functions
template <NetworkSolver nsolver>
void add_network_fun(std::map<NetworkFunKey_t, NetworkFun_t *> &network_functions) {
  network_functions[nsolver] = Ares::CalculateNetwork<nsolver>;
}

struct NucData {
  int num_species;
  std::array<int, NXNUC> na, nz, nn; /* atomic, proton and neutron number*/
  std::array<Real, NXNUC> m_ex, m, spin, gg, q,
      m_mol;                       /* Mass excess, mass, spin and partition function*/
  std::array<Real, NXNUC> prefact; /*Prefactor of the mass fraction equation*/
  std::array<std::array<Real, 24>, NXNUC> part;
  std::array<std::string, NXNUC> spec_name; /* name of the species/ isotope*/
};

} // namespace Ares

#endif // ARES_NETWORK_HPP_