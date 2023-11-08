//! \file white_dwarf.cpp
//! \brief Problem generator for a white dwarf.

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <iostream>
#include <sstream>

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>

// AthenaPK headers
#include "../main.hpp"

#include <singularity-eos/eos/eos.hpp>

namespace white_dwarf {
using namespace parthenon::driver::prelude;
using EOS = singularity::Variant<singularity::IdealGas, singularity::StellarCollapse,
                                 singularity::Helmholtz>;

extern char DATA_DIR[PATH_MAX];

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the Rayleigh-Taylor test

Real GRAV_NEWTON = 6.6743e-8;
Real POLY_CONST = 1.2435e15;
Real POLY_INDEX = 4. / 3;

Real RK4_solve(Real y_init, Real k1, Real k2, Real k3, Real k4, Real dy) {
  Real y_new = y_init + dy * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;

  return y_new;
}
// Computes shell mass assuming spherical symmetry
Real mass_funct(Real density, Real radius) {
  Real shell_mass = 4 * M_PI * SQR(radius) * density;

  return shell_mass;
}
// Solves ODE for mass
Real mass_solve(Real mass_init, Real density, Real radius, Real dr) {
  Real k1 = mass_funct(density, radius);
  Real k2 = mass_funct(density, radius + dr / 2.);
  Real k3 = mass_funct(density, radius + dr / 2.);
  Real k4 = mass_funct(density, radius + dr);

  Real new_mass = RK4_solve(mass_init, k1, k2, k3, k4, dr);

  return new_mass;
}
// Computes pressure assuming hydrostatic equilibrium
Real pres_funct(Real density, Real radius, Real mass) {
  Real shell_pressure = -1 * GRAV_NEWTON * mass * density / SQR(radius);

  return shell_pressure;
}
// Solves ODE for pressure
Real pres_solve(Real pres_init, Real density, Real radius, Real mass, Real dr) {
  Real k1 = pres_funct(density, radius, mass);
  Real k2 = pres_funct(density, radius + dr / 2., mass);
  Real k3 = pres_funct(density, radius + dr / 2., mass);
  Real k4 = pres_funct(density, radius + dr, mass);

  Real new_pres = RK4_solve(pres_init, k1, k2, k3, k4, dr);

  return new_pres;
}
// Computes density
Real dens_funct(Real pressure) {
  Real shell_density = pressure / POLY_CONST;
  shell_density = std::pow(shell_density, 1. / POLY_INDEX);

  return shell_density;
}

// Linear interpolation of densities
Real get_density_at_radius(Real radius, const std::array<Real, 948> &ic_radius,
                           const std::array<Real, 948> &ic_rho) {
  Real rho;

  // Find indicies of table elements to interpolate
  int i = 0;
  while (i < 948) {
    if ((radius > ic_radius[i]) && (radius < ic_radius[i + 1])) break;
    i++;
  }

  // Linear interpolation
  rho = ic_rho[i] + ((ic_rho[i + 1] - ic_rho[i]) / (ic_radius[i + 1] - ic_radius[i])) *
                        (radius - ic_radius[i]);
  return rho;
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // 1D WD profile generation
  // Takes in central density and mass composition
  Real density = pin->GetReal("problem/white_dwarf", "rho_central");
  Real rho_bgr = pin->GetOrAddReal("problem/white_dwarf", "rho_background", 1e0);
  Real temperature = pin->GetOrAddReal("problem/white_dwarf", "temp", 5e5);
  // Ignition temperature. If temperature is negative we don't ignite
  Real ign_temp = pin->GetOrAddReal("problem/white_dwarf", "ign_temp", 5e9);
  Real ign_radius = pin->GetOrAddReal("problem/white_dwarf", "ign_radius", 1e6);
  Real pressure = POLY_CONST * std::pow(density, POLY_INDEX);
  Real mass = 0.0;
  Real radius = 0.0;
  Real lambda_wd[3] = {13.714285714285715, 6.857142857142858, std::log10(temperature)};
  Real lambda_ign[3] = {13.714285714285715, 6.857142857142858, std::log10(ign_temp)};
  Real lambda_bcg[3] = {1.0, 1.0, std::log10(temperature)};

  const auto &eos = pmb->packages.Get("Hydro")->Param<singularity::EOS>("eos_host");

  std::string ic_data_file = "wd_ic.dat";
  const auto ic_file_str = "../data/" + ic_data_file;

  const int ic_cells = 948;
  std::array<Real, 948> ic_radius = {0.0};
  std::array<Real, 948> ic_rho = {0.0};

  std::fstream ic_file;
  ic_file.open(ic_file_str, std::ios::in);
  if (!ic_file) {
    PARTHENON_FAIL("Ares: Cannot open IC file")
  }

  std::string ic_instr;
  for (int i = 0; i < ic_cells; ++i) {
    getline(ic_file, ic_instr);
    std::istringstream in(ic_instr);
    in >> ic_radius[i];
    in >> ic_rho[i];
  }

  ic_file.close();

  Real wd_radius = ic_radius[947];

  /*
  int niter = 12;

  std::vector<Real> density_array;
  density_array.push_back(density);
  std::vector<Real> pressure_array;
  pressure_array.push_back(pressure);
  std::vector<Real> radius_array;
  radius_array.push_back(radius);

  int array_length = 0;

  Real dr = 4.0e5; // 4 kilometer initial step, 2000 km radius = 100 bins

  // RK4 solve for pressure
  while (density > 1.0e5) {
    radius += dr;
    radius_array.push_back(radius);

    array_length++;

    // eos_calc_ptgiven(&eos, pressure, xnuc, temperature, &density, &res);
    density = dens_funct(pressure);
    density_array.push_back(density);

    for (int i = 0; i < niter; i++) {
      Real error_mass = std::abs((mass_solve(mass, density, radius, dr) -
                                  mass_solve(mass, density, radius, dr / (2 * i))) /
                                 (mass_solve(mass, density, radius, dr) +
                                  mass_solve(mass, density, radius, dr / (2 * i))));
      Real error_pres =
          std::abs((pres_solve(pressure, density, radius, mass, dr) -
                    pres_solve(pressure, density, radius, mass, dr / (2 * i))) /
                   (pres_solve(pressure, density, radius, mass, dr) +
                    pres_solve(pressure, density, radius, mass, dr / (2 * i))));

      if ((error_mass > 1.e-5) || (error_pres > 1.e-5)) {
        dr /= 2 * i;
      } else {
        // RK4 step for mass
        mass = mass_solve(mass, density, radius, dr);
        // RK4 step for pressure
        pressure = pres_solve(pressure, density, radius, mass, dr);
        pressure_array.push_back(pressure);
        break;
      }
    }
  }
  */

  // Parthenon
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &v_dev = rc->Get("prim").data;
  auto &coords = pmb->coords;

  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  auto v = v_dev.GetHostMirrorAndCopy();

  // 1D to 3D profile mapping of generated white dwarf

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real radius =
            sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k)));

        if (radius < wd_radius) {
          Real u_d = get_density_at_radius(radius, ic_radius, ic_rho);
          u_d = (u_d > rho_bgr) ? u_d : rho_bgr;
          u(IDN, k, j, i) = u_d;
          if ((ign_temp > 0.0) && (radius < ign_radius)) {
            // Setup ignition spot
            u(IEN, k, j, i) =
                eos.InternalEnergyFromDensityTemperature(u_d, ign_temp, lambda_ign) * u_d;
          } else {
            u(IEN, k, j, i) =
                eos.InternalEnergyFromDensityTemperature(u_d, temperature, lambda_wd) *
                u_d;
          }
          // 50/50 CO mix in WD
          for (int is = NHYDRO; is < NHYDRO + NXNUC; is++) {
            u(is, k, j, i) = 0.0;
          }
          u(NHYDRO + 4, k, j, i) = u_d * 0.5;
          u(NHYDRO + 10, k, j, i) = u_d * 0.5;
        } else {
          u(IDN, k, j, i) = rho_bgr;
          u(IEN, k, j, i) = eos.InternalEnergyFromDensityTemperature(
                                u(IDN, k, j, i), temperature, lambda_bcg) *
                            rho_bgr;
          // Hydrogen background
          for (int is = NHYDRO; is < NHYDRO + NXNUC; is++) {
            u(is, k, j, i) = 0.0;
          }
          u(NHYDRO + 1, k, j, i) = rho_bgr * 1.0;
        }
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
      } // for i
    }   // for j
  }     // for k

  u_dev.DeepCopy(u);
  v_dev.DeepCopy(v);

  // eos_deinit(eos);

} // ProblemGenerator
} // namespace white_dwarf
