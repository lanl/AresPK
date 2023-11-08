#ifndef NSOLVERS_NETWORK_NSE_HPP_
#define NSOLVERS_NETWORK_NSE_HPP_

#include <cmath>

#include "../../main.hpp"
#include "../network.hpp"
#include "nsolvers.hpp"

const Real BOLTZMAN_CONSTANT = 1.3806504e-16;     /* g cm^2 / K s^2 */
const Real PLANCKS_CONSTANT_H = 6.62606896e-27;   /* g cm^2 / s */
const Real UNIFIED_ATOMIC_MASS = 1.660538782e-24; /* g */
const Real NUM_AVOGADRO = 6.02214199e23;          /* 1 / mol */
const int NSE_MAXITER = 500;
const Real NSE_DIFFVAR = 1e-12; /* variation for numerical derivatives */

template <>
struct Network<NetworkSolver::nse> {
  static KOKKOS_FORCEINLINE_FUNCTION void network_part(Real temp,
                                                       Ares::NucData *nucdata) {
    /* interpolates partition functions, given the temperature */
    int index, i;
    Real tempLeft, tempRight;
    Real dlgLeft, dlgRight;
    Real grad;

    static const Real network_parttemp[24] = {1.0e8, 1.5e8, 2.0e8, 3.0e8, 4.0e8, 5.0e8,
                                              6.0e8, 7.0e8, 8.0e8, 9.0e8, 1.0e9, 1.5e9,
                                              2.0e9, 2.5e9, 3.0e9, 3.5e9, 4.0e9, 4.5e9,
                                              5.0e9, 6.0e9, 7.0e9, 8.0e9, 9.0e9, 1.0e10};

    index = 0;
    temp = std::min(std::max(temp, network_parttemp[0]), network_parttemp[23]);

    while (temp > network_parttemp[index]) {
      index++;
    }
    if (index > 0) index--;

    tempLeft = network_parttemp[index];
    tempRight = network_parttemp[index + 1];

    for (i = 0; i < NXNUC; i++) {
      dlgLeft = nucdata->part[i][index];
      dlgRight = nucdata->part[i][index + 1];

      grad = (dlgRight - dlgLeft) / (tempRight - tempLeft);
      nucdata->gg[i] = exp(dlgLeft + (temp - tempLeft) * grad);
    }
  }

  static KOKKOS_FORCEINLINE_FUNCTION void calc_prefact(Real temp, Real rho,
                                                       Ares::NucData *nucdata) {
    for (int i = 0; i < NXNUC; i++) {
      nucdata->prefact[i] = nucdata->m[i] / rho * (2.0 * nucdata->spin[i] + 1.0) *
                            nucdata->gg[i] *
                            pow(2.0 * M_PI * nucdata->m[i] * BOLTZMAN_CONSTANT * temp /
                                    (PLANCKS_CONSTANT_H * PLANCKS_CONSTANT_H),
                                1.5);
    }
  }

  static KOKKOS_FORCEINLINE_FUNCTION void calc_yi(Real temp, Real rho, Real ye, Real mu_n,
                                                  Real mu_p, Real (&yi)[2], Real *x,
                                                  Ares::NucData nucdata) {
    int i;
    Real kt, xi;
    yi[0] = -1.0;
    yi[1] = 0.0;

    kt = 1.0 / (BOLTZMAN_CONSTANT * temp);
    for (i = 0; i < NXNUC; i++) {
      xi = nucdata.prefact[i] *
           exp(kt * (nucdata.nz[i] * mu_p + nucdata.nn[i] * mu_n + nucdata.q[i]));
      yi[0] += xi;
      yi[1] += UNIFIED_ATOMIC_MASS / nucdata.m[i] *
               ((ye - 1) * nucdata.nz[i] + ye * nucdata.nn[i]) * xi;

      if (x) x[i] = xi;
    }
  }

  static KOKKOS_FORCEINLINE_FUNCTION void guessmu(Real ye, Real *mu_n, Real *mu_p) {
    if (ye > 0.55) {
      *mu_n = -2.7e-5;
      *mu_p = -1.1e-6;
    } else if (ye > 0.53) {
      *mu_n = -0.22e-4;
      *mu_p = -0.06e-4;
    } else if (ye > 0.46) {
      *mu_n = -0.19e-4;
      *mu_p = -0.08e-4;
    } else if (ye > 0.44) {
      *mu_n = -0.145e-4;
      *mu_p = -0.135e-4;
    } else if (ye > 0.42) {
      *mu_n = -0.11e-4;
      *mu_p = -0.18e-4;
    } else {
      *mu_n = -0.08e-4;
      *mu_p = -0.22e-4;
    }
  }

  static KOKKOS_FORCEINLINE_FUNCTION void Solve(Real temp, Real rho, Real ye,
                                                VariablePack<Real> &cons,
                                                Ares::NucData nucdata, const int i,
                                                const int j, const int k, const Real dt) {
    int iter, ii;
    Real mu_n, mu_p, dmu_n, dmu_p;
    Real y[2], y2[2];
    Real jac[4]; /* order: a11, a12, a21, a22 */
    Real det, kt;
    Real xsum, nn, ne;

    /* preparations */
    network_part(temp, &nucdata); // Partition function
    calc_prefact(temp, rho, &nucdata);
    guessmu(ye, &mu_n, &mu_p); // Initial Guess for the chemical potentials

    /* Newton-Raphson method for root finding to solve for mu_n and mu_p */
    iter = 0;
    while (iter < NSE_MAXITER) {
      calc_yi(temp, rho, ye, mu_n, mu_p, y, NULL, nucdata);
      if (fmax(fabs(y[0]), fabs(y[1])) < 1e-12) break;

      dmu_n = fabs(mu_n) * NSE_DIFFVAR;
      if (dmu_n == 0.0) dmu_n = NSE_DIFFVAR;
      calc_yi(temp, rho, ye, mu_n + dmu_n, mu_p, y2, NULL, nucdata);
      for (ii = 0; ii < 2; ii++)
        jac[ii * 2] = (y2[ii] - y[ii]) / dmu_n;

      dmu_p = fabs(mu_p) * NSE_DIFFVAR;
      if (dmu_p == 0.0) dmu_p = NSE_DIFFVAR;
      calc_yi(temp, rho, ye, mu_n, mu_p + dmu_p, y2, NULL, nucdata);
      for (ii = 0; ii < 2; ii++)
        jac[ii * 2 + 1] = (y2[ii] - y[ii]) / dmu_p;

      det = 1.0 / (jac[0] * jac[3] - jac[1] * jac[2]);
      dmu_n = det * (y[0] * jac[3] - y[1] * jac[1]);
      dmu_p = det * (y[1] * jac[0] - y[0] * jac[2]);

      mu_n -= dmu_n;
      mu_p -= dmu_p;
      iter++;
    }
    kt = 1.0 / (BOLTZMAN_CONSTANT * temp);
    Real x_i[NXNUC];
    Real d_energy_density = 0.0;

    for (ii = 0; ii < NXNUC; ii++) {
      x_i[ii] = rho * nucdata.prefact[ii] *
                exp(kt * (nucdata.nz[ii] * mu_p + nucdata.nn[ii] * mu_n + nucdata.q[ii]));

      d_energy_density -= nucdata.q[ii] * (x_i[ii] - cons(ii + NHYDRO, k, j, i)) *
                          NUM_AVOGADRO / nucdata.m_mol[ii];
      cons(ii + NHYDRO, k, j, i) = x_i[ii];
    }
    cons(IEN, k, j, i) += d_energy_density; // Updating the energy
  }
};
#endif // NSOLVERS_NETWORK_NSE_HPP_
