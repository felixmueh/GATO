#pragma once
#include "cpu_strip_common.hpp"
#if __has_include("cpu_strip_ic.hpp")
#include "cpu_strip_ic.hpp"
#endif
#if __has_include("cpu_strip_sgs.hpp")
#include "cpu_strip_sgs.hpp"
#endif
#if __has_include("cpu_strip_schwarz.hpp")
#include "cpu_strip_schwarz.hpp"
#endif
#if __has_include("cpu_strip_jacobi.hpp")
#include "cpu_strip_jacobi.hpp"
#endif
#if __has_include("cpu_strip_amg.hpp") && __has_include(<amgcl/amg.hpp>)
#include "cpu_strip_amg.hpp"
#endif

namespace upstream_cpu {
inline std::unique_ptr<Method> make(const std::string& name) {
#if __has_include("cpu_strip_ic.hpp")
  if (name == "incomplete_cholesky") return std::make_unique<IncompleteCholesky>();
#endif
#if __has_include("cpu_strip_sgs.hpp")
  if (name == "block_sgs") return std::make_unique<BlockSgs>();
#endif
#if __has_include("cpu_strip_schwarz.hpp")
  if (name == "schwarz") return std::make_unique<Schwarz>();
#endif
#if __has_include("cpu_strip_jacobi.hpp")
  if (name == "block_jacobi") return std::make_unique<BlockJacobi>();
#endif
#if __has_include("cpu_strip_amg.hpp") && __has_include(<amgcl/amg.hpp>)
  if (name == "amg") return std::make_unique<Amg>();
#endif
  throw std::invalid_argument("Unavailable CPU strip preconditioner: " + name);
}
inline bool available(const std::string& name) {
  if (name == "cpu_native_control" || name == "symmetric_stair" ||
      name == "polynomial2" || name == "polynomial3") return true;
  try { (void)make(name); return true; } catch (const std::invalid_argument&) { return false; }
}
}  // namespace upstream_cpu
