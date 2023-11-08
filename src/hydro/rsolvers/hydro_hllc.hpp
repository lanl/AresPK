//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hllc.cpp
//  \brief HLLC Riemann solver for hydrodynamics
//
//  Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
// - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
// - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

#ifndef RSOLVERS_HYDRO_HLLC_HPP_
#define RSOLVERS_HYDRO_HLLC_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"
#include "rsolvers.hpp"

#include <singularity-eos/eos/eos.hpp>

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//  \brief The HLLC Riemann solver for hydrodynamics (adiabatic)
template <>
struct Riemann<Fluid::euler, RiemannSolver::hllc, NetworkSolver::none> {
  static KOKKOS_FORCEINLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const ScratchPad2D<Real> &ifl, const ScratchPad2D<Real> &ifr,
        const singularity::EOS &eos, const int num_species,
        VariablePack<Real> &eos_lambda) {
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    static constexpr Real C_LIGHT = 2.99792458e10;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[(NHYDRO)], wri[(NHYDRO)];
      Real fl[(NHYDRO)], fr[(NHYDRO)], flxi[(NHYDRO)];
      //--- Step 1.  Load L/R states into local variables
      wli[IDN] = wl(IDN, i);
      wli[IV1] = wl(ivx, i);
      wli[IV2] = wl(ivy, i);
      wli[IV3] = wl(ivz, i);
      wli[IPR] = wl(IPR, i);

      wri[IDN] = wr(IDN, i);
      wri[IV1] = wr(ivx, i);
      wri[IV2] = wr(ivy, i);
      wri[IV3] = wr(ivz, i);
      wri[IPR] = wr(IPR, i);

      Real &abar = eos_lambda(0, k, j, i);
      Real &zbar = eos_lambda(1, k, j, i);
      Real &lT = eos_lambda(2, k, j, i);
      Real lambda[3] = {abar, zbar, lT};

      //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.6)
      // Compute adiabatic indices at the interfaces
      Real gm1l, gm1r, gammal, gammar, bulkmodl, bulkmodr, igm1l, igm1r;
      gammal = ifl(0, i);
      gammar = ifr(0, i);
      bulkmodl = gammal * wli[IPR];
      bulkmodr = gammar * wri[IPR];
      gm1l = ifl(1, i) - 1;
      gm1r = ifr(1, i) - 1;
      igm1l = 1.0 / gm1l;
      igm1r = 1.0 / gm1r;

      Real cl = C_LIGHT *
                std::sqrt(gammal / (1 + (igm1l + wli[IDN] * SQR(C_LIGHT) / wli[IPR])));
      Real cr = C_LIGHT *
                std::sqrt(gammar / (1 + (igm1r + wri[IDN] * SQR(C_LIGHT) / wri[IPR])));

      Real el = wli[IPR] * igm1l +
                0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
      Real er = wri[IPR] * igm1r +
                0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
      Real rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
      Real ca = .5 * (cl + cr);               // average sound speed
      Real pstar = .5 * (wli[IPR] + wri[IPR]) - .5 * (wri[IV1] - wli[IV1]) * rhoa * ca;

      //--- Step 3.  Wave speed estimates
      Real ql, qr;
      Real sl, sr, sstar;
      ql = (pstar <= wli[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gammal + 1) / (2 * gammar) * (pstar / wli[IPR] - 1.0));
      qr = (pstar <= wri[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gammal + 1) / (2 * gammar) * (pstar / wri[IPR] - 1.0));

      sl = wli[IV1] - cl * ql;
      sr = wri[IV1] + cr * qr;

      //-- Step 4. Compute the max/min wave speeds based on L/R
      Real bp = sr > 0.0 ? sr : (TINY_NUMBER);
      Real bm = sl < 0.0 ? sl : (TINY_NUMBER);

      //-- Step 5. Compute L/R conserved variables
      Real vxl = wli[IV1] - sl;
      Real vxr = wri[IV1] - sr;

      Real tl = wli[IPR] + vxl * wli[IDN] * wli[IV1];
      Real tr = wri[IPR] + vxr * wri[IDN] * wri[IV1];

      Real ml = wli[IDN] * vxl;
      Real mr = -(wri[IDN] * vxr);

      // Determine the contact wave speed...
      sstar = (tl - tr) / (ml + mr);
      // ...and the pressure at the contact surface
      Real cp = (ml * tr + mr * tl) / (ml + mr);
      cp = cp > 0.0 ? cp : 0.0;

      //--- Step 6. Compute L/R fluxes along the line bm, bp
      vxl = wli[IV1] - bm;
      vxr = wri[IV1] - bp;

      fl[IDN] = wli[IDN] * vxl;
      fr[IDN] = wri[IDN] * vxr;

      fl[IV1] = wli[IDN] * wli[IV1] * vxl + wli[IPR];
      fr[IV1] = wri[IDN] * wri[IV1] * vxr + wri[IPR];

      fl[IV2] = wli[IDN] * wli[IV2] * vxl;
      fr[IV2] = wri[IDN] * wri[IV2] * vxr;

      fl[IV3] = wli[IDN] * wli[IV3] * vxl;
      fr[IV3] = wri[IDN] * wri[IV3] * vxr;

      fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
      fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

      //-- Step 7. Compute HLLC flux
      Real ur, ul, ustar;
      if (sstar >= 0.0) {
        ul = sstar / (sstar - bm);
        ur = 0.0;
        ustar = -bm / (sstar - bm);
      } else {
        ul = 0.0;
        ur = -sstar / (bp - sstar);
        ustar = bp / (bp - sstar);
      }

      flxi[IDN] = ul * fl[IDN] + ur * fr[IDN];
      flxi[IV1] = ul * fl[IV1] + ur * fr[IV1] + ustar * cp;
      flxi[IV2] = ul * fl[IV2] + ur * fr[IV2];
      flxi[IV3] = ul * fl[IV3] + ur * fr[IV3];
      flxi[IEN] = ul * fl[IEN] + ur * fr[IEN] + ustar * cp * sstar;

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
    });
  }
};

template <>
struct Riemann<Fluid::euler, RiemannSolver::hllc, NetworkSolver::nse> {
  static KOKKOS_FORCEINLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const ScratchPad2D<Real> &ifl, const ScratchPad2D<Real> &ifr,
        const singularity::EOS &eos, const int num_species,
        VariablePack<Real> &eos_lambda) {
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    int varcount = NHYDRO + num_species;
    static constexpr Real C_LIGHT = 2.99792458e10;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[(NHYDRO)], wri[(NHYDRO)];
      Real fl[(NHYDRO)], fr[(NHYDRO)], flxi[(NHYDRO)];
      //--- Step 1.  Load L/R states into local variables
      wli[IDN] = wl(IDN, i);
      wli[IV1] = wl(ivx, i);
      wli[IV2] = wl(ivy, i);
      wli[IV3] = wl(ivz, i);
      wli[IPR] = wl(IPR, i);

      wri[IDN] = wr(IDN, i);
      wri[IV1] = wr(ivx, i);
      wri[IV2] = wr(ivy, i);
      wri[IV3] = wr(ivz, i);
      wri[IPR] = wr(IPR, i);

      Real &abar = eos_lambda(0, k, j, i);
      Real &zbar = eos_lambda(1, k, j, i);
      Real &lT = eos_lambda(2, k, j, i);
      Real lambda[3] = {abar, zbar, lT};

      //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.6)
      // Compute adiabatic indices at the interfaces
      Real gm1l, gm1r, gammal, gammar, bulkmodl, bulkmodr, igm1l, igm1r;
      gammal = ifl(0, i);
      gammar = ifr(0, i);
      bulkmodl = gammal * wli[IPR];
      bulkmodr = gammar * wri[IPR];
      gm1l = ifl(1, i) - 1;
      gm1r = ifr(1, i) - 1;
      igm1l = 1.0 / gm1l;
      igm1r = 1.0 / gm1r;

      Real cl = C_LIGHT *
                std::sqrt(gammal / (1 + (igm1l + wli[IDN] * SQR(C_LIGHT) / wli[IPR])));
      Real cr = C_LIGHT *
                std::sqrt(gammar / (1 + (igm1r + wri[IDN] * SQR(C_LIGHT) / wri[IPR])));

      Real el = wli[IPR] * igm1l +
                0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
      Real er = wri[IPR] * igm1r +
                0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
      Real rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
      Real ca = .5 * (cl + cr);               // average sound speed
      Real pstar = .5 * (wli[IPR] + wri[IPR]) - .5 * (wri[IV1] - wli[IV1]) * rhoa * ca;

      //--- Step 3.  Wave speed estimates
      Real ql, qr;
      Real sl, sr, sstar;
      ql = (pstar <= wli[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gammal + 1) / (2 * gammar) * (pstar / wli[IPR] - 1.0));
      qr = (pstar <= wri[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gammal + 1) / (2 * gammar) * (pstar / wri[IPR] - 1.0));

      sl = wli[IV1] - cl * ql;
      sr = wri[IV1] + cr * qr;

      //-- Step 4. Compute the max/min wave speeds based on L/R
      Real bp = sr > 0.0 ? sr : (TINY_NUMBER);
      Real bm = sl < 0.0 ? sl : -(TINY_NUMBER);

      //-- Step 5. Compute L/R conserved variables
      Real vxl = wli[IV1] - sl;
      Real vxr = wri[IV1] - sr;

      Real tl = wli[IPR] + vxl * wli[IDN] * wli[IV1];
      Real tr = wri[IPR] + vxr * wri[IDN] * wri[IV1];

      Real ml = wli[IDN] * vxl;
      Real mr = -(wri[IDN] * vxr);

      // Determine the contact wave speed...
      sstar = (tl - tr) / (ml + mr);
      // ...and the pressure at the contact surface
      Real cp = (ml * tr + mr * tl) / (ml + mr);
      cp = cp > 0.0 ? cp : 0.0;

      //--- Step 6. Compute L/R fluxes along the line bm, bp
      vxl = wli[IV1] - bm;
      vxr = wri[IV1] - bp;

      fl[IDN] = wli[IDN] * vxl;
      fr[IDN] = wri[IDN] * vxr;

      fl[IV1] = wli[IDN] * wli[IV1] * vxl + wli[IPR];
      fr[IV1] = wri[IDN] * wri[IV1] * vxr + wri[IPR];

      fl[IV2] = wli[IDN] * wli[IV2] * vxl;
      fr[IV2] = wri[IDN] * wri[IV2] * vxr;

      fl[IV3] = wli[IDN] * wli[IV3] * vxl;
      fr[IV3] = wri[IDN] * wri[IV3] * vxr;

      fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
      fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

      //-- Step 7. Compute HLLC flux
      Real ur, ul, ustar;
      if (sstar >= 0.0) {
        ul = sstar / (sstar - bm);
        ur = 0.0;
        ustar = -bm / (sstar - bm);
      } else {
        ul = 0.0;
        ur = -sstar / (bp - sstar);
        ustar = bp / (bp - sstar);
      }

      flxi[IDN] = ul * fl[IDN] + ur * fr[IDN];
      flxi[IV1] = ul * fl[IV1] + ur * fr[IV1] + ustar * cp;
      flxi[IV2] = ul * fl[IV2] + ur * fr[IV2];
      flxi[IV3] = ul * fl[IV3] + ur * fr[IV3];
      flxi[IEN] = ul * fl[IEN] + ur * fr[IEN] + ustar * cp * sstar;

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
      for (int is = NHYDRO; is < varcount; is++) {
        cons.flux(ivx, is, k, j, i) =
            ul * wl(is, i) * wli[IDN] * vxl + ur * wr(is, i) * wri[IDN] * vxr;
      }
    });
  }
};

#endif // RSOLVERS_HYDRO_HLLC_HPP_