#ifndef NSOLVERS_NSOLVERS_HPP_
#define NSOLVERS_NSOLVERS_HPP_

#include "../../main.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

// First declare general template
template <NetworkSolver nsolver>
struct Network;

// Now include the specializations
#include "network_nse.hpp"

template <>
struct Network<NetworkSolver::none> {
  static KOKKOS_FORCEINLINE_FUNCTION void Solve(Real temp, Real rho, Real ye,
                                                VariablePack<Real> &cons,
                                                Ares::NucData nucdata, const int i,
                                                const int j, const int k, const Real dt) {
    ((void)0);
  }
};

#endif // NSOLVERS_NSOLVERS_HPP_