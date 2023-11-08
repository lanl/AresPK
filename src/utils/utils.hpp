#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <parthenon/package.hpp>

#include "../main.hpp"

// Plog header
#include <plog/Log.h>

using namespace parthenon;

namespace utils {
bool ShouldLog(int partition, bool log_per_process = false,
               bool log_per_partition = false);

std::string TaskInfo(int i = -1);

std::string MeshInfo(MeshData<Real> *u);

TaskStatus SumMass(MeshData<Real> *u, std::vector<Real> *reduce_sum);
TaskStatus BinMasses(MeshData<Real> *u, std::vector<Real> *bins, std::vector<Real> *CoM);
} // namespace utils
#endif // UTILS_HPP_