#ifndef GSOLVERS_GRAVITATION_CONSTANT_HPP_
#define GSOLVERS_GRAVITATION_CONSTANT_HPP_

// Ares headers
#include "../main.hpp"
#include "gsolvers.hpp"
#include "parthenon/package.hpp"

using parthenon::Real;
using parthenon::VariablePack;

template <>
class Gravitation<GravitySolver::constant> {
 public:
  static KOKKOS_INLINE_FUNCTION void Solve(VariablePack<Real> &prim,
                                           VariablePack<Real> &cons, const int i,
                                           const int j, const int k, Real grav_accel) {
    const Real den = prim(IDN, k, j, i);

    cons(IM1, k, j, i) -= den * grav_accel;
    cons(IEN, k, j, i) -= den * prim(IV1, k, j, i) * grav_accel;
  };
};

#endif // GSOLVERS_GRAVITATION_CONSTANT_HPP_
