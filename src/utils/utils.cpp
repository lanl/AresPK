#include <parthenon/package.hpp>

#include <cmath>

#include "../main.hpp"

// Plog header
#include <plog/Log.h>

using namespace parthenon;

namespace utils {
bool ShouldLog(int partition, bool log_per_process = false,
               bool log_per_partition = false) {
  return ((partition == 0 || log_per_partition) &&
          (Globals::my_rank == 0 || log_per_process));
}

std::string TaskInfo(int i = -1) {
  if (i == -1) {
    return "[Process " + std::to_string(Globals::my_rank) + "] ";
  } else {
    return "[Process " + std::to_string(Globals::my_rank) + ", Partition " +
           std::to_string(i) + "] ";
  }
}

std::string MeshInfo(MeshData<Real> *u) {
  return "[MeshData with bounds: i[" +
         std::to_string(u->GetBoundsI(IndexDomain::entire).s) + ", " +
         std::to_string(u->GetBoundsI(IndexDomain::entire).e) + "], j[" +
         std::to_string(u->GetBoundsJ(IndexDomain::entire).s) + ", " +
         std::to_string(u->GetBoundsJ(IndexDomain::entire).e) + "], k[" +
         std::to_string(u->GetBoundsK(IndexDomain::entire).s) + ", " +
         std::to_string(u->GetBoundsK(IndexDomain::entire).e) + "] containing " +
         std::to_string(u->NumBlocks()) + " blocks] ";
}

struct MassBins {
  using value_type = double[];

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  const int value_count;
  const Real step, xcenter, ycenter, zcenter;
  MeshBlockPack<VariablePack<Real>> X_;

  MassBins(const MeshBlockPack<VariablePack<Real>> &X, const int nbins, Real r_edge,
           Real x, Real y, Real z)
      : value_count(nbins), X_(X), xcenter(x), ycenter(y), zcenter(z),
        step(r_edge / (nbins - 1)) {}

  KOKKOS_INLINE_FUNCTION void operator()(const int b, const int k, const int j,
                                         const int i, value_type sum) const {
    const int ndim = X_.GetNdim();
    const auto &cons = X_(b);
    const auto &coords = cons.GetCoords(b);
    const Real dens = cons(IDN, k, j, i);
    const Real dx = coords.Dxc<X1DIR>();
    Real mass = dens;
    for (int x = 0; x < ndim; ++x) {
      mass *= dx;
    }
    const Real radius = std::sqrt(std::pow(coords.Xc<1>(k, j, i) - xcenter, 2) +
                                  std::pow(coords.Xc<2>(k, j, i) - ycenter, 2) +
                                  std::pow(coords.Xc<3>(k, j, i) - zcenter, 2));
    // Clamp to num_bins - 1 for cells outside r_edge
    const int rad_idx = std::min(int(radius / step), value_count - 1);
    if (rad_idx < 0) {
      PARTHENON_FAIL("Ares: Gravity rad_idx negative. No mass left.")
    }
    sum[rad_idx] += mass;
  }

  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    for (int i = 0; i < value_count; ++i) {
      dst[i] += src[i];
    }
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    for (int i = 0; i < value_count; ++i) {
      sum[i] = 0.0;
    }
  }
};

TaskStatus BinMasses(MeshData<Real> *u, std::vector<Real> *bins, std::vector<Real> *CoM) {
  auto pmb = u->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &cons_pack = u->PackVariables(std::vector<std::string>{"cons"});
  const int num_bins = hydro_pkg->Param<int>("num_bins");

  Kokkos::View<double *> mass_bins("sums", num_bins);
  Kokkos::parallel_reduce(
      "MassBins",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, 1}),
      MassBins(cons_pack, num_bins, (*CoM)[4], (*CoM)[0], (*CoM)[1], (*CoM)[2]),
      mass_bins);
  Kokkos::View<double *>::HostMirror mass_binsh = Kokkos::create_mirror_view(mass_bins);
  Kokkos::deep_copy(mass_binsh, mass_bins);
  std::string log_string = "";
  for (int i = 0; i < num_bins; ++i) {
    (*bins)[i] += mass_binsh[i];
    log_string += std::to_string(mass_binsh[i]) + " ";
  }
  PLOG(plog::verbose) << TaskInfo() << "Bins: " << log_string;
  return TaskStatus::complete;
} // BinMasses

struct MassSum {
  using value_type = Real[];

  // Tell Kokkos the result array's number of entries.
  // This must be a public value in the functor.
  const int value_count;

  const Real dens_thresh;

  MeshBlockPack<VariablePack<Real>> X_;

  // Be sure to set value_count in the constructor.
  MassSum(const MeshBlockPack<VariablePack<Real>> &X, const Real dens)
      : value_count(5), X_(X), dens_thresh(dens) {}

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void operator()(const int b, const int k, const int j,
                                         const int i, value_type sum) const {
    const int ndim = X_.GetNdim();
    const auto &cons = X_(b);
    const auto &coords = cons.GetCoords(b);
    const Real dx = coords.Dxc<X1DIR>();
    const Real dens = cons(IDN, k, j, i);
    Real mass = dens;
    for (int x = 0; x < ndim; ++x) {
      mass *= dx;
    }
    sum[0] += mass * coords.Xc<1>(k, j, i);
    sum[1] += mass * coords.Xc<2>(k, j, i);
    sum[2] += mass * coords.Xc<3>(k, j, i);
    sum[3] += mass;
    const Real dv = coords.CellVolume(k, j, i);
    if (dens > dens_thresh) {
      sum[4] += dv;
    }
  }

  // value_type here is already a "reference" type,
  // so we don't pass it in by reference here.
  KOKKOS_INLINE_FUNCTION void join(value_type dst, const value_type src) const {
    dst[0] += src[0];
    dst[1] += src[1];
    dst[2] += src[2];
    dst[3] += src[3];
    dst[4] += src[4];
  }

  KOKKOS_INLINE_FUNCTION void init(value_type sum) const {
    sum[0] = 0.0;
    sum[1] = 0.0;
    sum[2] = 0.0;
    sum[3] = 0.0;
    sum[4] = 0.0;
  }
};

TaskStatus SumMass(MeshData<Real> *u, std::vector<Real> *reduce_sum) {
  auto pmb = u->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &cons_pack = u->PackVariables(std::vector<std::string>{"cons"});
  const int ndim = cons_pack.GetNdim();
  const Real dens_thresh = hydro_pkg->Param<Real>("dens_thresh");

  Real mass_total[5];
  Kokkos::parallel_reduce("SumMassFull",
                          Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
                              DevExecSpace(), {0, kb.s, jb.s, ib.s},
                              {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
                              {1, 1, 1, 1}),
                          MassSum(cons_pack, dens_thresh), mass_total);
  for (int i = 0; i < 5; ++i) {
    mass_total[i] += (*reduce_sum)[i];
  }
  *reduce_sum = std::vector<Real>(mass_total, mass_total + 5);
  PLOG(plog::verbose) << TaskInfo() << MeshInfo(u)
                      << "CoM: X=" << mass_total[0] / mass_total[3]
                      << " Y=" << mass_total[1] / mass_total[3]
                      << " Z=" << mass_total[2] / mass_total[3]
                      << " Mass=" << mass_total[3] << " Volume=" << mass_total[4];
  return TaskStatus::complete;
} // SumMass

} // namespace utils