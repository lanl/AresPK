//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle.cpp
//  \brief HLLE Riemann solver for hydrodynamics
//
//  Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//  very diffusive, especially for contacts, and so it is not recommended for use in
//  applications.  However, as shown by Einfeldt et al.(1991), it is positively
//  conservative (cannot return negative densities or pressure), so it is a useful
//  option when other approximate solvers fail and/or when extra dissipation is needed.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
// - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
// - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

#ifndef RSOLVERS_HYDRO_HLLE_HPP_
#define RSOLVERS_HYDRO_HLLE_HPP_

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
//  \brief The HLLE Riemann solver for hydrodynamics (adiabatic)
template <>
struct Riemann<Fluid::euler, RiemannSolver::hlle, NetworkSolver::none> {
  KOKKOS_FORCEINLINE_FUNCTION void static Solve(
      parthenon::team_mbr_t const &member, const int k, const int j, const int il,
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

      //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)
      Real al, ar, el, er;

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

      el = wli[IPR] * igm1l +
           0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
      er = wri[IPR] * igm1r +
           0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
      Real rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
      Real ca = .5 * (cl + cr);               // average sound speed
      Real pmid = .5 * (wli[IPR] + wri[IPR] + (wli[IV1] - wri[IV1]) * rhoa * ca);
      Real umid = .5 * (wli[IV1] + wri[IV1] + (wli[IPR] - wri[IPR]) / (rhoa * ca));
      Real rhol = wli[IDN] + (wli[IV1] - umid) * rhoa / ca; // mid-left density
      Real rhor = wri[IDN] + (umid - wri[IV1]) * rhoa / ca; // mid-right density

      //--- Step 3.  Compute sound speed in L,R
      Real ql, qr;
      ql = (pmid <= wli[IPR])
               ? 1.0
               : (1.0 + (gammal + 1) / std::sqrt(2 * gammal) * (pmid / wli[IPR] - 1.0));
      qr = (pmid <= wri[IPR])
               ? 1.0
               : (1.0 + (gammar + 1) / std::sqrt(2 * gammar) * (pmid / wri[IPR] - 1.0));

      //--- Step 4. Compute the max/min wave speeds based on L/R states

      al = wli[IV1] - cl * ql;
      ar = wri[IV1] + cr * qr;

      Real bp = ar > 0.0 ? ar : 0.0;
      Real bm = al < 0.0 ? al : 0.0;

      //-- Step 5. Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = wli[IV1] - bm;
      Real vxr = wri[IV1] - bp;

      fl[IDN] = wli[IDN] * vxl;
      fr[IDN] = wri[IDN] * vxr;

      fl[IV1] = wli[IDN] * wli[IV1] * vxl;
      fr[IV1] = wri[IDN] * wri[IV1] * vxr;

      fl[IV2] = wli[IDN] * wli[IV2] * vxl;
      fr[IV2] = wri[IDN] * wri[IV2] * vxr;

      fl[IV3] = wli[IDN] * wli[IV3] * vxl;
      fr[IV3] = wri[IDN] * wri[IV3] * vxr;

      fl[IV1] += wli[IPR];
      fr[IV1] += wri[IPR];
      fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
      fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

      //--- Step 6. Compute the HLLE flux at interface.
      Real tmp = 0.0;
      if (bp != bm) tmp = 0.5 * (bp + bm) / (bp - bm);

      flxi[IDN] = 0.5 * (fl[IDN] + fr[IDN]) + (fl[IDN] - fr[IDN]) * tmp;
      flxi[IV1] = 0.5 * (fl[IV1] + fr[IV1]) + (fl[IV1] - fr[IV1]) * tmp;
      flxi[IV2] = 0.5 * (fl[IV2] + fr[IV2]) + (fl[IV2] - fr[IV2]) * tmp;
      flxi[IV3] = 0.5 * (fl[IV3] + fr[IV3]) + (fl[IV3] - fr[IV3]) * tmp;
      flxi[IEN] = 0.5 * (fl[IEN] + fr[IEN]) + (fl[IEN] - fr[IEN]) * tmp;

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
    });
  }
};

template <>
struct Riemann<Fluid::euler, RiemannSolver::hlle, NetworkSolver::nse> {
  KOKKOS_FORCEINLINE_FUNCTION void static Solve(
      parthenon::team_mbr_t const &member, const int k, const int j, const int il,
      const int iu, const int ivx, const ScratchPad2D<Real> &wl,
      const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
      const ScratchPad2D<Real> &ifl, const ScratchPad2D<Real> &ifr,
      const singularity::EOS &eos, const int num_species,
      VariablePack<Real> &eos_lambda) {
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int varcount = NHYDRO + num_species;
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

      //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)
      Real al, ar, el, er;

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

      el = wli[IPR] * igm1l +
           0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
      er = wri[IPR] * igm1r +
           0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
      Real rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
      Real ca = .5 * (cl + cr);               // average sound speed
      Real pmid = .5 * (wli[IPR] + wri[IPR] + (wli[IV1] - wri[IV1]) * rhoa * ca);
      Real umid = .5 * (wli[IV1] + wri[IV1] + (wli[IPR] - wri[IPR]) / (rhoa * ca));
      Real rhol = wli[IDN] + (wli[IV1] - umid) * rhoa / ca; // mid-left density
      Real rhor = wri[IDN] + (umid - wri[IV1]) * rhoa / ca; // mid-right density

      //--- Step 3.  Compute sound speed in L,R
      Real ql, qr;
      ql = (pmid <= wli[IPR])
               ? 1.0
               : (1.0 + (gammal + 1) / std::sqrt(2 * gammal) * (pmid / wli[IPR] - 1.0));
      qr = (pmid <= wri[IPR])
               ? 1.0
               : (1.0 + (gammar + 1) / std::sqrt(2 * gammar) * (pmid / wri[IPR] - 1.0));

      //--- Step 4. Compute the max/min wave speeds based on L/R states

      al = wli[IV1] - cl * ql;
      ar = wri[IV1] + cr * qr;

      Real bp = ar > 0.0 ? ar : 0.0;
      Real bm = al < 0.0 ? al : 0.0;

      //-- Step 5. Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = wli[IV1] - bm;
      Real vxr = wri[IV1] - bp;

      fl[IDN] = wli[IDN] * vxl;
      fr[IDN] = wri[IDN] * vxr;

      fl[IV1] = wli[IDN] * wli[IV1] * vxl;
      fr[IV1] = wri[IDN] * wri[IV1] * vxr;

      fl[IV2] = wli[IDN] * wli[IV2] * vxl;
      fr[IV2] = wri[IDN] * wri[IV2] * vxr;

      fl[IV3] = wli[IDN] * wli[IV3] * vxl;
      fr[IV3] = wri[IDN] * wri[IV3] * vxr;

      fl[IV1] += wli[IPR];
      fr[IV1] += wri[IPR];
      fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
      fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

      for (int is = NHYDRO; is < varcount; is++) {
        fl[is] = wli[is] * vxl;
        fr[is] = wri[is] * vxr;
      }

      //--- Step 6. Compute the HLLE flux at interface.
      Real tmp = 0.0;
      if (bp != bm) tmp = 0.5 * (bp + bm) / (bp - bm);

      flxi[IDN] = 0.5 * (fl[IDN] + fr[IDN]) + (fl[IDN] - fr[IDN]) * tmp;
      flxi[IV1] = 0.5 * (fl[IV1] + fr[IV1]) + (fl[IV1] - fr[IV1]) * tmp;
      flxi[IV2] = 0.5 * (fl[IV2] + fr[IV2]) + (fl[IV2] - fr[IV2]) * tmp;
      flxi[IV3] = 0.5 * (fl[IV3] + fr[IV3]) + (fl[IV3] - fr[IV3]) * tmp;
      flxi[IEN] = 0.5 * (fl[IEN] + fr[IEN]) + (fl[IEN] - fr[IEN]) * tmp;
      for (int is = NHYDRO; is < varcount; is++) {
        flxi[is] = 0.5 * (fl[is] + fr[is]) + (fl[is] - fr[is]) * tmp;
      }

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
      for (int is = NHYDRO; is < varcount; is++) {
        cons.flux(ivx, is, k, j, i) =
            0.5 * (wl(is, i) * wli[IDN] * vxl + wr(is, i) * wri[IDN] * vxr) +
            (wl(is, i) * wli[IDN] * vxl - wr(is, i) * wri[IDN] * vxr) * tmp;
      }
    });
  }
};

#endif // RSOLVERS_HYDRO_HLLE_HPP_
