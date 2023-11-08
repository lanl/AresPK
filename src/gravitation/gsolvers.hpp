#ifndef GSOLVERS_GSOLVERS_HPP_
#define GSOLVERS_GSOLVERS_HPP_

// Ares headers
#include "../main.hpp"
#include "parthenon/package.hpp"
#include <cmath>

namespace gravitation {
using parthenon::Coordinates_t;
using parthenon::Real;
using parthenon::VariablePack;

// First declare general template
template <GravitySolver gsolver>
struct Gravitation;

// Empty gravitation
template <>
struct Gravitation<GravitySolver::none> {
  KOKKOS_INLINE_FUNCTION void Solve(VariablePack<Real> &prim, VariablePack<Real> &cons,
                                    Coordinates_t coords, const int i, const int j,
                                    const int k, const Real dt) const {}
};

template <>
struct Gravitation<GravitySolver::monopole> {
  void set_parameters(std::vector<Real> *CoM, std::vector<Real> *mass_bins,
                      Real newton_G) {
    Kokkos::View<Real *, Kokkos::HostSpace> CoM_host(CoM->data(), CoM->size());
    this->CoM = Kokkos::View<Real *>("CoM", CoM->size());
    Kokkos::deep_copy(this->CoM, CoM_host);
    Kokkos::View<Real *, Kokkos::HostSpace> mass_bins_host(mass_bins->data(),
                                                           mass_bins->size());
    this->mass_bins = Kokkos::View<Real *>("MassBins", mass_bins->size());
    Kokkos::deep_copy(this->mass_bins, mass_bins_host);
    this->num_bins = mass_bins->size();
    this->newton_G = newton_G;
  }

  KOKKOS_INLINE_FUNCTION void Solve(VariablePack<Real> &prim, VariablePack<Real> &cons,
                                    Coordinates_t coords, const int i, const int j,
                                    const int k, const Real dt) const {
    // Center of mass coordinates
    const Real xcm = CoM[0];
    const Real ycm = CoM[1];
    const Real zcm = CoM[2];

    // Define geometry objects
    const Real radius2 = SQR(coords.Xc<1>(i) - xcm) + SQR(coords.Xc<2>(j) - ycm) +
                         SQR(coords.Xc<3>(k) - zcm);

    const Real den = prim(IDN, k, j, i);

    // Define components of gravity
    const Real radius = std::sqrt(radius2);
    const int rad_idx = std::min(int((num_bins - 1) * radius / CoM[4]), num_bins - 1);

    Real grav_accel =
        (radius < 1e-10) ? 0.0 : -1.0 * newton_G * mass_bins[rad_idx] / radius2;
    // Real grav_accel = -1.0;
    const Real grav_x = grav_accel * (coords.Xc<1>(i) - xcm) / radius;
    const Real grav_y = grav_accel * (coords.Xc<2>(j) - ycm) / radius;
    const Real grav_z = grav_accel * (coords.Xc<3>(k) - zcm) / radius;

    cons(IM1, k, j, i) += den * grav_x * dt;
    cons(IM2, k, j, i) += den * grav_y * dt;
    cons(IM3, k, j, i) += den * grav_z * dt;
    cons(IEN, k, j, i) += den *
                          (prim(IV1, k, j, i) * grav_x + prim(IV2, k, j, i) * grav_y +
                           prim(IV3, k, j, i) * grav_z) *
                          dt;
  }

 private:
  Kokkos::View<Real *> CoM;
  Kokkos::View<Real *> mass_bins;
  int num_bins;
  Real newton_G;
};

template <>
struct Gravitation<GravitySolver::constant> {
  void set_parameters(Real grav_accel) { this->grav_accel = grav_accel; };

  KOKKOS_INLINE_FUNCTION void Solve(VariablePack<Real> &prim, VariablePack<Real> &cons,
                                    Coordinates_t coords, const int i, const int j,
                                    const int k, const Real dt) const {
    const Real den = prim(IDN, k, j, i);

    cons(IM1, k, j, i) += den * (-1.0 * grav_accel) * dt;
    cons(IEN, k, j, i) += den * prim(IV1, k, j, i) * (-1.0 * grav_accel) * dt;
  }

 private:
  Real grav_accel;
};

} // namespace gravitation

#endif // GSOLVERS_GSOLVERS_HPP_
